import base64
from collections import Counter
import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import uuid

import cv2
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from ultralytics import YOLO

app = FastAPI()

model = YOLO("yolov8n.pt")
SUPPORTED_VIDEO_FORMATS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
MAX_RETURNED_DETECTIONS = 500
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_STORED_FRAMES = 300
MAX_QUESTION_FRAMES = 12
VIDEO_DATA_DIR = Path(os.getenv("VIDEO_DATA_DIR", "/data/videos"))


@app.get("/")
def home():
    return {"status": "Vision Service is running"}


def _read_video_metadata(video_id: str) -> dict:
    if not re.fullmatch(r"[a-f0-9]{32}", video_id):
        raise HTTPException(status_code=404, detail="Video not found.")
    metadata_path = VIDEO_DATA_DIR / video_id / "analysis.json"
    if not metadata_path.is_file():
        raise HTTPException(status_code=404, detail="Video not found.")
    with metadata_path.open(encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


@app.post("/analyze_video")
async def analyze_video(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_VIDEO_FORMATS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported video format. Supported extensions: {', '.join(sorted(SUPPORTED_VIDEO_FORMATS))}",
        )

    video_id = uuid.uuid4().hex
    video_dir = VIDEO_DATA_DIR / video_id
    frame_dir = video_dir / "frames"
    original_path = video_dir / f"original{suffix}"
    capture = None
    completed = False
    try:
        frame_dir.mkdir(parents=True, exist_ok=False)
        uploaded_bytes = 0
        with original_path.open("wb") as video_file:
            while chunk := await file.read(1024 * 1024):
                uploaded_bytes += len(chunk)
                if uploaded_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Video uploads must be 200 MB or smaller.")
                video_file.write(chunk)

        capture = cv2.VideoCapture(str(original_path))
        if not capture.isOpened():
            raise HTTPException(
                status_code=422,
                detail="The video could not be decoded. Check that its codec is supported.",
            )

        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        total_frames_estimate = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_interval = max(1, round(fps))
        duration_estimate = total_frames_estimate / fps if total_frames_estimate > 0 else 0
        storage_interval_seconds = max(1, math.ceil(duration_estimate / MAX_STORED_FRAMES))
        storage_interval_frames = max(sample_interval, round(fps * storage_interval_seconds))

        frame_number = 0
        sampled_frames = 0
        stored_frames = []
        detections_by_class = Counter()
        detections = []
        total_detections = 0
        class_names = model.names
        if class_names is None:
            raise RuntimeError("The object detector did not provide class names.")

        while True:
            success, frame = capture.read()
            if not success:
                break

            timestamp_seconds = round(frame_number / fps, 2)
            if frame_number % storage_interval_frames == 0:
                preview_frame = frame
                height, width = preview_frame.shape[:2]
                scale = min(1.0, 768 / max(width, height))
                if scale < 1:
                    preview_frame = cv2.resize(
                        preview_frame,
                        (round(width * scale), round(height * scale)),
                    )
                encoded, jpeg = cv2.imencode(
                    ".jpg",
                    preview_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 75],
                )
                if not encoded:
                    raise HTTPException(status_code=500, detail="Could not save a sampled video frame.")
                frame_name = f"{len(stored_frames):04d}.jpg"
                (frame_dir / frame_name).write_bytes(jpeg.tobytes())
                stored_frames.append({
                    "index": len(stored_frames),
                    "time_seconds": timestamp_seconds,
                    "file": f"frames/{frame_name}",
                })

            if frame_number % sample_interval == 0:
                sampled_frames += 1
                prediction_results = model.predict(frame, verbose=False)
                if not prediction_results:
                    raise RuntimeError("The object detector returned no prediction result.")
                result = prediction_results[0]
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls.item())
                        label = class_names[class_id]
                        confidence = round(float(box.conf.item()), 4)
                        detections_by_class[label] += 1
                        total_detections += 1
                        if len(detections) < MAX_RETURNED_DETECTIONS:
                            detections.append({
                                "time_seconds": timestamp_seconds,
                                "label": label,
                                "confidence": confidence,
                            })
            frame_number += 1

        if frame_number == 0:
            raise HTTPException(status_code=422, detail="The uploaded video contains no readable frames.")

        metadata = {
            "video_id": video_id,
            "filename": filename,
            "duration_seconds": round(frame_number / fps, 2),
            "total_frames": frame_number,
            "sampled_frames": sampled_frames,
            "sample_interval_seconds": round(sample_interval / fps, 2),
            "detections_by_class": dict(detections_by_class),
            "supported_classes": list(class_names.values()),
            "total_detections": total_detections,
            "detections": detections,
            "detections_truncated": total_detections > len(detections),
            "model_note": "YOLOv8n COCO detects general objects; it is not trained to identify helmets or safety violations.",
            "stored_frame_interval_seconds": round(storage_interval_frames / fps, 2),
            "stored_frames": stored_frames,
        }
        with (video_dir / "analysis.json").open("w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file)
        completed = True
        return metadata
    finally:
        if capture is not None:
            capture.release()
        await file.close()
        if not completed:
            shutil.rmtree(video_dir, ignore_errors=True)


@app.get("/videos")
def list_videos():
    videos = []
    if VIDEO_DATA_DIR.exists():
        for metadata_path in VIDEO_DATA_DIR.glob("*/analysis.json"):
            with metadata_path.open(encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
            videos.append({
                "video_id": metadata["video_id"],
                "filename": metadata["filename"],
                "duration_seconds": metadata["duration_seconds"],
                "stored_frames": len(metadata["stored_frames"]),
            })
    return {"videos": sorted(videos, key=lambda video: video["filename"].lower())}


@app.get("/videos/{video_id}")
def get_video(video_id: str):
    return _read_video_metadata(video_id)


@app.get("/videos/{video_id}/frames")
def get_question_frames(
    video_id: str,
    count: int = Query(default=MAX_QUESTION_FRAMES, ge=1, le=MAX_QUESTION_FRAMES),
    query: str = "",
):
    metadata = _read_video_metadata(video_id)
    frames = metadata["stored_frames"]
    if not frames:
        raise HTTPException(status_code=404, detail="No sampled frames are available.")

    matching_labels = {
        label.lower()
        for label in metadata["detections_by_class"]
        if label.lower() in query.lower()
    }
    if matching_labels:
        preferred_timestamps = {
            detection["time_seconds"]
            for detection in metadata["detections"]
            if detection["label"].lower() in matching_labels
        }
        candidates = [
            frame for frame in frames
            if frame["time_seconds"] in preferred_timestamps
        ]
    else:
        candidates = frames

    if not candidates:
        candidates = frames
    if len(candidates) > count:
        selected_positions = [
            round(index * (len(candidates) - 1) / (count - 1))
            for index in range(count)
        ] if count > 1 else [len(candidates) // 2]
        candidates = [candidates[position] for position in selected_positions]

    encoded_frames = []
    video_dir = VIDEO_DATA_DIR / video_id
    for frame in candidates:
        frame_path = video_dir / frame["file"]
        if not frame_path.is_file():
            continue
        encoded_frames.append({
            "time_seconds": frame["time_seconds"],
            "image_base64": base64.b64encode(frame_path.read_bytes()).decode("ascii"),
        })
    if not encoded_frames:
        raise HTTPException(status_code=404, detail="Stored sampled frames could not be read.")
    return {"video_id": video_id, "filename": metadata["filename"], "frames": encoded_frames}


@app.post("/detect_stream")
def simulate_detection(video_path: str = "sample.mp4"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    event = {
        "timestamp": timestamp,
        "location": "Zone A - Warehouse",
        "object_detected": "person",
        "violation": "Missing Hard Hat / PPE Alert",
        "confidence": 0.89
    }
    return {"events_logged": [event]}
