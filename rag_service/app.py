import json
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Literal
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import chromadb
import requests
from sentence_transformers import SentenceTransformer

app = FastAPI(title="RAG Vector Memory Service")

# Initialize ChromaDB persistent client pointing to a local directory
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="safety_events")

# Load a lightweight embedding model locally
model = SentenceTransformer("all-MiniLM-L6-v2")
VISION_SERVICE_URL = os.getenv("VISION_SERVICE_URL", "http://vision-service:8001")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
SEARCH_HISTORY_DB = Path(os.getenv("SEARCH_HISTORY_DB", "/data/rag/search_history.db"))


@contextmanager
def _connect_search_history() -> Iterator[sqlite3.Connection]:
    SEARCH_HISTORY_DB.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(SEARCH_HISTORY_DB, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def _initialize_search_history() -> None:
    with _connect_search_history() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                created_at TEXT NOT NULL,
                matched_events_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            )
            """
        )


@app.on_event("startup")
def initialize_search_history() -> None:
    _initialize_search_history()

class EventLog(BaseModel):
    event_id: str
    timestamp: str
    location: str
    description: str

class EventBatch(BaseModel):
    events: list[EventLog] = Field(min_length=1, max_length=500)

class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)

class VideoQuestion(BaseModel):
    video_id: str = Field(min_length=1, max_length=64)
    query: str = Field(min_length=1, max_length=2000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=8)

@app.get("/")
def home():
    return {"status": "RAG Service is active and memory is ready!"}

@app.post("/add_event")
def add_event(event: EventLog):
    # Create text embedding for the event description
    text_to_embed = f"At {event.timestamp}, in {event.location}: {event.description}"
    vector = model.encode(text_to_embed).tolist()
    
    # Store into ChromaDB vector database
    collection.add(
        documents=[text_to_embed],
        embeddings=[vector],
        metadatas=[{"timestamp": event.timestamp, "location": event.location}],
        ids=[event.event_id]
    )
    return {"message": "Event successfully vectorized and stored in RAG memory!", "event_id": event.event_id}

@app.post("/add_events")
def add_events(batch: EventBatch):
    texts_to_embed = [
        f"At {event.timestamp}, in {event.location}: {event.description}"
        for event in batch.events
    ]
    vectors = model.encode(texts_to_embed).tolist()
    collection.upsert(
        documents=texts_to_embed,
        embeddings=vectors,
        metadatas=[
            {"timestamp": event.timestamp, "location": event.location}
            for event in batch.events
        ],
        ids=[event.event_id for event in batch.events],
    )
    return {
        "message": "Events successfully vectorized and stored in RAG memory!",
        "events_stored": len(batch.events),
    }

@app.get("/search")
def search_events(query: str):
    normalized_query = query.strip()
    if not normalized_query:
        raise HTTPException(status_code=422, detail="Enter a search query.")

    # Embed user's natural language question
    query_vector = model.encode(normalized_query).tolist()
    
    # Search vector database for top matching events
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=3
    )

    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []
    created_at = datetime.now(timezone.utc).isoformat()
    with _connect_search_history() as connection:
        cursor = connection.execute(
            """
            INSERT INTO search_history (
                query, created_at, matched_events_json, metadata_json
            ) VALUES (?, ?, ?, ?)
            """,
            (
                normalized_query,
                created_at,
                json.dumps(documents),
                json.dumps(metadatas),
            ),
        )
        history_id = cursor.lastrowid

    return {
        "history_id": history_id,
        "query": normalized_query,
        "created_at": created_at,
        "matched_events": documents,
        "metadata": metadatas,
    }


@app.get("/search_history")
def get_search_history(limit: int = Query(default=50, ge=1, le=200)):
    with _connect_search_history() as connection:
        rows = connection.execute(
            """
            SELECT id, query, created_at, matched_events_json, metadata_json
            FROM search_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return {
        "searches": [
            {
                "id": row["id"],
                "query": row["query"],
                "created_at": row["created_at"],
                "matched_events": json.loads(row["matched_events_json"]),
                "metadata": json.loads(row["metadata_json"]),
            }
            for row in rows
        ]
    }


def _is_complex_query(query: str) -> bool:
    complex_patterns = (
        r"\bwhy\b",
        r"\bcompare\b",
        r"\bdescribe\b",
        r"\bwhat happens\b",
        r"\bwhat is happening\b",
        r"\bhow does\b",
        r"\bwhen\b|\bwhat time\b",
        r"\bat what time\b",
        r"\bbefore\b",
        r"\bafter\b",
        r"\bcolor\b|\bcolour\b",
        r"\bwearing\b",
        r"\bdoing\b",
        r"\bexplain\b",
        r"\bsummarize\b|\bsummarise\b",
    )
    return any(re.search(pattern, query, re.IGNORECASE) for pattern in complex_patterns)


def _fetch_video_metadata(video_id: str):
    video_response = requests.get(
        f"{VISION_SERVICE_URL}/videos/{quote(video_id, safe='')}",
        timeout=(10, 30),
    )
    video_response.raise_for_status()
    return video_response.json()


def _fetch_video_frames(video_id: str, query: str, count: int):
    frames_response = requests.get(
        f"{VISION_SERVICE_URL}/videos/{quote(video_id, safe='')}/frames",
        params={"count": count, "query": query},
        timeout=(10, 60),
    )
    frames_response.raise_for_status()
    return frames_response.json()["frames"]


def _select_evenly_spaced_frames(frames: list[dict], count: int) -> list[dict]:
    if len(frames) <= count:
        return frames
    positions = [
        round(index * (len(frames) - 1) / (count - 1))
        for index in range(count)
    ] if count > 1 else [len(frames) // 2]
    return [frames[position] for position in positions]


def _build_video_prompt(
    query: str,
    video: dict,
    frames: list[dict],
    history: list[ChatTurn],
) -> str:
    timestamps = ", ".join(f"{frame['time_seconds']}s" for frame in frames)
    is_count_question = bool(
        re.search(r"\b(how many|count|number of)\b", query, re.IGNORECASE)
    )
    counts = (
        ", ".join(
            f"{label}: {count} detections"
            for label, count in sorted(video["detections_by_class"].items())
        ) or "No objects detected by the general-object detector."
    ) if is_count_question else "Not included; the current question is not about counts."
    prior_questions = [
        f"user: {turn.content}" for turn in history if turn.role == "user"
    ]
    conversation = "\n".join(prior_questions) or "(No previous questions.)"
    return (
        "Answer the user's question about this analyzed video using only the supplied "
        "sampled frames and video metadata. Be clear about uncertainty and say when "
        "the evidence is insufficient. Write one or two complete sentences that "
        "directly answer the current question; never return only timestamps or labels. "
        "If the frames do not answer it, state that in words. Add a timestamp after "
        "the answer, not instead of one. "
        "Do not infer identity, intent, or safety/PPE "
        "compliance. Cite relevant video timestamps in [MM:SS] format. Object "
        "detection totals count detections across frames and do not necessarily count "
        "unique people or objects. Answer the current question rather than repeating an "
        "earlier answer. Use earlier user questions only to resolve follow-up references. "
        "Do not give object counts unless the current question asks for a count. "
        "For questions about actions, describe only actions visible in the supplied "
        "frames; if they do not establish an action, say so instead of repeating an "
        "earlier count. Verify factual claims against this video's frames and metadata.\n\n"
        f"Video: {video['filename']}\n"
        f"Duration: {video['duration_seconds']} seconds\n"
        f"Object detections across sampled frames: {counts}\n"
        f"Frames are provided in timestamp order: {timestamps}\n\n"
        f"Previous conversation:\n{conversation}\n\n"
        f"Question: {query}"
    )


def _ask_ollama(prompt: str, frames: list[dict]) -> str:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [{
                "role": "user",
                "content": prompt,
                "images": [frame["image_base64"] for frame in frames],
            }],
            "stream": False,
            "options": {"temperature": 0.2, "num_ctx": 8192},
        },
        timeout=(10, 300),
    )
    response.raise_for_status()
    answer = response.json().get("message", {}).get("content", "").strip()
    if not answer:
        raise ValueError("The local vision model returned an empty answer.")
    return answer


def _ask_gemini(prompt: str, frames: list[dict]) -> str:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured.")
    parts: list[dict] = [{"text": prompt}]
    for frame in frames:
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": frame["image_base64"],
            }
        })
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{quote(GEMINI_MODEL, safe='.-')}:generateContent",
        params={"key": GEMINI_API_KEY},
        json={
            "contents": [{"parts": parts}],
            "generationConfig": {"temperature": 0.2},
        },
        timeout=(10, 180),
    )
    response.raise_for_status()
    candidates = response.json().get("candidates", [])
    answer = "".join(
        part.get("text", "")
        for part in candidates[0].get("content", {}).get("parts", [])
        if part.get("text")
    ).strip() if candidates else ""
    if not answer:
        raise ValueError("Gemini returned an empty answer.")
    return answer


def _object_count_answer(query: str, video: dict):
    if not re.search(r"\b(how many|count|number of)\b", query, re.IGNORECASE):
        return None
    normalized_query = query.lower()
    if re.search(r"\b(people|persons?|humans?)\b", normalized_query):
        labels = ["person"] if "person" in video["supported_classes"] else []
    else:
        labels = [
            label
            for label in video["supported_classes"]
            if re.search(rf"\b{re.escape(label.lower())}s?\b", normalized_query)
        ]
    if not labels:
        return None
    label = labels[0]
    count = video["detections_by_class"].get(label, 0)
    answer = (
        f"The object detector recorded {count} {label} detections across sampled frames "
        f"in {video['filename']}."
    )
    if label == "person":
        answer += " This is not a count of unique people."
    timestamps = []
    if re.search(r"\b(when|time|timestamp)\b", normalized_query):
        timestamps = sorted({
            detection["time_seconds"]
            for detection in video["detections"]
            if detection["label"] == label
        })[:20]
        if timestamps:
            answer += " Detected at approximately: " + ", ".join(
                f"{timestamp}s" for timestamp in timestamps
            ) + "."
    return answer, timestamps


@app.post("/ask")
def ask_about_video(question: VideoQuestion):
    query = question.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Enter a question before searching.")

    try:
        video = _fetch_video_metadata(question.video_id)
    except requests.RequestException as error:
        raise HTTPException(
            status_code=502,
            detail=f"Could not load the analyzed video metadata: {error}",
        ) from error

    direct_answer = _object_count_answer(query, video)
    if direct_answer:
        return {
            "answer": direct_answer[0],
            "provider": "YOLOv8n detector",
            "model": "yolov8n",
            "frame_timestamps": direct_answer[1],
        }

    prefer_cloud = _is_complex_query(query) and bool(GEMINI_API_KEY)
    failures = []
    cloud_frames = []
    if prefer_cloud:
        try:
            cloud_frames = _fetch_video_frames(question.video_id, query, count=12)
            if not cloud_frames:
                raise ValueError("No sampled frames are available.")
            prompt = _build_video_prompt(query, video, cloud_frames, question.history)
            answer = _ask_gemini(prompt, cloud_frames)
            return {
                "answer": answer,
                "provider": "Gemini",
                "model": GEMINI_MODEL,
                "frame_timestamps": [frame["time_seconds"] for frame in cloud_frames],
                "routing": "complex-query preference",
            }
        except (requests.RequestException, ValueError) as error:
            failures.append(_provider_failure("Gemini", error))
            local_frames = _select_evenly_spaced_frames(cloud_frames, count=2)
    else:
        try:
            local_frames = _fetch_video_frames(question.video_id, query, count=2)
        except requests.RequestException as error:
            raise HTTPException(
                status_code=502,
                detail=f"Could not load sampled video frames: {error}",
            ) from error

    if local_frames:
        try:
            prompt = _build_video_prompt(query, video, local_frames, question.history)
            answer = _ask_ollama(prompt, local_frames)
            return {
                "answer": answer,
                "provider": "Ollama",
                "model": OLLAMA_MODEL,
                "frame_timestamps": [frame["time_seconds"] for frame in local_frames],
                "routing": "local fallback" if prefer_cloud else "local-first",
            }
        except (requests.RequestException, ValueError) as error:
            failures.append(_provider_failure("Ollama", error))
    else:
        failures.append("No sampled frames are available.")

    if GEMINI_API_KEY and not prefer_cloud:
        try:
            cloud_frames = _fetch_video_frames(question.video_id, query, count=12)
            if not cloud_frames:
                raise ValueError("No sampled frames are available.")
            prompt = _build_video_prompt(query, video, cloud_frames, question.history)
            answer = _ask_gemini(prompt, cloud_frames)
            return {
                "answer": answer,
                "provider": "Gemini",
                "model": GEMINI_MODEL,
                "frame_timestamps": [frame["time_seconds"] for frame in cloud_frames],
                "routing": "cloud fallback",
            }
        except (requests.RequestException, ValueError) as error:
            failures.append(_provider_failure("Gemini", error))

    raise HTTPException(
        status_code=503,
        detail="No configured vision model could answer this question. " + "; ".join(failures),
    )


def _provider_failure(provider: str, error: Exception) -> str:
    if isinstance(error, requests.HTTPError) and error.response is not None:
        return f"{provider} returned HTTP {error.response.status_code}"
    return f"{provider} request failed ({type(error).__name__})"