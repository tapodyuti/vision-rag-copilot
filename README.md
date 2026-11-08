# Vision-RAG Safety Copilot

A Docker Compose application with a Streamlit interface, a YOLO-based video analysis service, and a vector-search service for safety-event records.

## Run the application

From the project directory, build and start the services:

```powershell
docker compose up --build -d
```

Open the application at <http://localhost:8501>.

The first build can take several minutes because it installs the Python and machine-learning dependencies. On first startup, the RAG service downloads its embedding model and Ollama downloads the local vision model; these downloads can take several minutes. Model caches are kept in Docker volumes. Check service status and logs if startup is still in progress:

```powershell
docker compose ps
docker compose logs -f ollama-model-init vision-service rag-service frontend
```

### Configure Gemini cloud fallback

The local Ollama model works without a cloud key. To enable automatic Gemini fallback for complex questions, copy [.env.example](./.env.example) to `.env` and add a Google Gemini API key:

```powershell
Copy-Item .env.example .env
notepad .env
```

Keep `.env` private; never commit or share the API key. Restart after changing it:

```powershell
docker compose up --build -d
```

Simple object-count questions use the detector's recorded counts. Other questions use local Ollama first; questions classified as more complex prefer Gemini when a key is configured, and try the other provider if the preferred one fails. This is a practical routing heuristic, not a guarantee that a model is objectively best. The result names the model used.

Stop the services with:

```powershell
docker compose down
```

## Upload and analyze a video

1. Open <http://localhost:8501>.
2. Under **Analyze an uploaded video**, choose a video file.
3. Review the video preview and select **Analyze video**.
4. Review the duration, sampled-frame count, object counts, and timestamped detections. The interface also confirms whether the findings were added to searchable memory.

The upload limit is 200 MB. The interface accepts `.avi`, `.m4v`, `.mkv`, `.mov`, `.mp4`, `.mpeg`, `.mpg`, and `.webm` files. Successful analysis also depends on whether OpenCV can decode the video's codec.

The analyzer samples approximately one frame per second. Processing time depends on video duration and the computer running Docker.

## What the video analysis does—and does not do

Video analysis uses the bundled YOLOv8n model, which recognizes general objects from the COCO dataset. It reports detected object labels, confidence values, and approximate timestamps.

**It does not detect helmets, identify PPE violations, or determine whether a person is following safety rules. Do not use its output as a safety-compliance decision.** A model trained and validated for the required PPE classes and conditions is needed for that.

After a successful analysis, detected general-object labels, counts, and sampled timestamps are added to RAG search. These entries are observations from the COCO model—not PPE or safety-violation findings.

Uploaded originals and timestamped sampled frames are retained in a Docker volume. The analyzer stores up to 300 representative frames per video, while object detection continues to sample about once per second. Local Ollama Q&A examines two evenly spaced relevant frames; Gemini Q&A examines up to 12. Long videos or events between sampled frames may be missed.

In **Ask about a video**, choose an analyzed video and ask questions in natural language; follow-up turns are kept per video in the current Streamlit session. The Q&A and event-search panel selection stays in place after submitting a question. Local Ollama examines two selected frames per question; Gemini examines up to 12. Earlier user questions provide follow-up context, while factual answers are grounded in the current frames and detector metadata. Answers may be incomplete, especially for events outside the selected frames. Audio is not transcribed. Exact person counts mean detections across frames, not unique individuals. When Gemini is selected, the question and sampled frames are sent to Google; the UI displays a notice. Cloud use can incur charges and is subject to Google's terms and privacy policies. Without Gemini configured, questions use local Ollama only.

Uploaded videos are retained in the `video_data` Docker volume across restarts and `docker compose down`. They are not sent to Gemini as full videos—only the selected sampled frames are sent if Gemini is used.

## Services

| Service | Purpose | Local address |
| --- | --- | --- |
| `frontend` | Streamlit user interface | <http://localhost:8501> |
| `vision-service` | Video uploads and object detection | <http://localhost:8001> |
| `rag-service` | Embeddings and safety-event vector search | <http://localhost:8002> |
| `ollama` | Local Qwen2.5-VL model (internal Docker network) | — |
| `redis-broker` | Redis service used by the Compose setup | <localhost:6379> |

The frontend talks to backend services over HTTP on the Docker Compose network. Ollama runs locally in its own container. The RAG service keeps the existing text-vector search for event records and offers a separate video-question endpoint that supplies timestamped frames to the selected vision model.

The RAG API supports `POST /ask` for video Q&A, `POST /add_events` for a batch of up to 500 event records, `GET /search?query=...` for semantic text search, and `GET /search_history` for saved searches and result snapshots. Open `http://localhost:8002/docs` for interactive API documentation. The vision API provides `GET /videos` and `GET /videos/{video_id}` for retained analyses.

Open the **Search history** panel next to **Ask about a video** and **Search event records** to review past searches. Searches made in **Search event records** are saved with their top matching results in the persistent `rag_data` Docker volume. The history panel displays the 50 most recent searches; choosing one shows its saved results without rerunning the query. History is local to this Compose setup and shared by anyone using this app. It is retained until the `rag_data` volume is removed; `docker compose down -v` removes that volume along with other application data.

## Video analysis API

The vision service accepts a multipart upload at `POST /analyze_video`. For example, from PowerShell:

```powershell
curl.exe -F "file=@C:\path\to\video.mp4" http://localhost:8001/analyze_video
```

The response includes the filename, duration, number of frames sampled, counts by detected object class, and up to 500 timestamped detections. `detections_truncated` indicates whether additional detections were omitted from the response.

## Troubleshooting

- **The browser cannot connect:** Confirm that Docker Desktop is running, then run `docker compose ps`. The frontend should publish port `8501`.
- **A service exited:** Inspect its logs with `docker compose logs --tail 100 <service-name>`.
- **Local Q&A is unavailable:** Check `docker compose logs ollama ollama-model-init rag-service`. The first run downloads the local vision model and the RAG embedding model; allow startup to finish before asking a question. The frontend waits for the RAG service to become healthy.
- **Gemini is not used:** Confirm `GEMINI_API_KEY` is set in `.env`, then recreate the services with `docker compose up --build -d`.
- **A video is missing from the picker:** Upload and analyze it again. Only videos analyzed after persistent storage was added appear there.
- **Q&A misses an event:** Local Ollama receives two selected frames; Gemini receives up to 12. A maximum of 300 representative frames are retained per video, and important details between sampled frames may not be represented. Audio is not transcribed.
- **The first video request takes a while:** The vision service may be downloading its model or initializing its inference runtime. Check `docker compose logs -f vision-service`.
- **The uploaded video is rejected or has no readable frames:** Check the 200 MB limit, use a supported file extension, and try a video encoded with a codec supported by OpenCV.
- **Objects are missing from the results:** Detection is performed on approximately one frame per second, and the general-purpose model may not recognize every object in every scene.

## Project layout

```text
frontend/        Streamlit interface
vision_service/  FastAPI video-analysis service and YOLO model
rag_service/     FastAPI embedding and ChromaDB search service
video_data       Docker volume with retained originals and sampled frames
ollama_data      Docker volume with the local vision model
huggingface_cache Docker volume with the RAG embedding model
rag_data         Docker volume with persistent search history and result snapshots
docker-compose.yml
.env.example     Optional cloud-model configuration template
```

## Development

Rebuild and restart a service after changing its code or dependencies:

```powershell
docker compose up --build -d frontend vision-service rag-service
```

The vision and RAG services expose their FastAPI documentation at `/docs` on their respective local ports. `docker compose down` preserves named volumes; `docker compose down -v` deletes named volumes, including retained videos, model data, and saved search history.
