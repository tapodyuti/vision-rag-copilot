# Project Handoff Notes

Use this file to bring a new assistant session up to speed on this project. The project is `vision-rag-copilot`, a Docker Compose application for video object analysis and safety-event search.

## Current app

- **Frontend:** Streamlit at <http://localhost:8501>
- **Vision API:** FastAPI at <http://localhost:8001>
- **RAG API:** FastAPI at <http://localhost:8002>
- **Local vision model:** Ollama container with Qwen2.5-VL 3B
- **Optional cloud vision model:** Google Gemini, enabled by `GEMINI_API_KEY`
- **Redis:** Compose service at port 6379
- Start or restart the stack from the project directory:

  ```powershell
  docker compose up --build -d
  ```

- Check containers and logs:

  ```powershell
  docker compose ps
  docker compose logs --tail 100 frontend vision-service rag-service
  ```

- Stop the stack without deleting the project:

  ```powershell
  docker compose down
  ```

  Docker Compose containers can be started again later with `docker compose up -d`.

## Work completed

1. Fixed frontend API URLs in `frontend/ui.py`: Docker-internal calls use HTTP, not HTTPS.
2. Fixed RAG dependency compatibility in `rag_service/requirements.txt`:
   - Pinned NumPy to `1.26.4` for compatibility with the current ChromaDB.
   - Pinned Transformers to `4.36.2` and Hugging Face Hub to `0.20.3` for compatibility with the current Sentence Transformers and PyTorch.
3. Added multipart upload support to `vision_service/requirements.txt` with `python-multipart==0.0.9`.
4. Added `POST /analyze_video` to `vision_service/main.py`. It accepts supported video formats, enforces a 200 MB upload limit, analyzes about one frame per second with YOLOv8n, returns object counts and timestamped detections, and deletes its temporary upload after analysis.
5. Added a video picker, preview, analysis button, and results display to `frontend/ui.py`.
6. Wrote `README.md` for setup, usage, API, and troubleshooting, and `CHANGELOG.md` to summarize project changes.
7. Added batch ingestion to the RAG service and automatically save object labels, counts, and sampled timestamps after video analysis so they can be searched.
8. Added persistent video/frame storage, video Q&A, and local Ollama/Gemini provider routing. Ollama receives up to two selected frames per question; Gemini receives up to 12. Gemini receives sampled frames only and is disclosed in the UI.
9. Replaced the Q&A/search tabs with a persistent panel selector and a standard form for video questions. Follow-up prompts include previous user questions as context while requiring answers to be grounded in current frames and metadata.
10. Added a RAG health check so the frontend waits for the embedding model to finish loading before it starts.
11. Added SQLite-backed search history in the persistent `rag_data` Docker volume. Each semantic search saves its query, timestamp, and result snapshot; a dedicated Search history panel shows the 50 most recent and can reopen their results without rerunning.

## Video analysis details and limitations

- Accepted extensions: `.avi`, `.m4v`, `.mkv`, `.mov`, `.mp4`, `.mpeg`, `.mpg`, `.webm`.
- Uploads are limited to 200 MB. Video decoding also depends on the codec being supported by OpenCV.
- The current YOLOv8n model is a **general-object COCO model**, not a helmet/PPE detector. Its output must not be treated as a safety-compliance decision.
- Uploaded videos and up to 300 representative sampled frames per video are retained in the Docker `video_data` volume.
- Video Q&A uses selected frames and detector metadata; local Ollama receives up to two frames per question and Gemini up to 12. The UI keeps per-video conversation history for follow-ups; previous user questions help resolve references, while prior assistant answers are excluded from model context. The Q&A/search panel selection persists across reruns. Answers are not guaranteed to describe every visual detail. Object counts are detections across sampled frames, not unique entities.
- Simple count questions are answered from detector counts. Other questions are routed by a query heuristic: Ollama first for typical queries and Gemini first for queries classified as complex when `GEMINI_API_KEY` is configured, with provider fallback on request errors.
- When Gemini answers, question text and up to 12 sampled frames are sent to Google; full video files are not sent. The UI discloses cloud use. API key is configured through an untracked local `.env`.
- RAG entries from video analysis are general-object observations, not PPE detection or safety-violation decisions.
- Event-record searches and their result snapshots persist across sessions in the shared local `rag_data` volume; the dedicated Search history panel lists the 50 most recent. History is retained until the named volume is deleted.
- The existing `/detect_stream` endpoint remains a mock/demo response; it does not read a live camera stream or the user's uploaded video.
- First start downloads the YOLO model, RAG embedding model, and Qwen2.5-VL model through Ollama; allow time for this.

## Verification already performed

- `docker compose config -q` succeeded.
- The frontend returned HTTP 200 on port 8501.
- Vision and RAG service startup/health were checked.
- A generated test MP4 was uploaded to `/analyze_video` and returned HTTP 200 with analysis results.
- An unsupported `.txt` upload returned HTTP 415 as expected.
- Confirmed the frontend waits until the RAG container reports healthy before starting, then used the browser to ask a count question and a follow-up. Both answers rendered, the follow-up avoided repeating the count, and switching to event search and back preserved the Q&A transcript.
- A visual-action follow-up reported that the sampled frames did not show enough detail to determine the action; this is an evidence limitation, not a chat submission failure.
- After the conversational prompt update, a test video returned the expected 19 person detections and a local Ollama follow-up response with the frame timestamp; the response was brief, so complex visual interpretation remains model-limited.

## Possible next steps

- If helmet/PPE detection is needed, select and validate an appropriate PPE-trained model and update the detector and interface claims accordingly.
- Replace the mock `/detect_stream` endpoint if actual camera or live-stream processing is required.
- Consider adding automated tests for video upload, invalid input, and analysis-result handling.

## Instructions for a future assistant

Read this file and the current source before making changes. Preserve existing project changes. Do not claim the app detects PPE or safety violations unless a suitable model has actually been integrated and validated. After changes, run the relevant checks and report what was verified.
