# Changelog

Notable project changes are recorded here.

## 2026-11-08

### Added

- Added persistent storage for uploaded videos and timestamped representative frames.
- Added video question answering with local Ollama vision and Gemini cloud routing, plus visible cloud-use disclosure in the UI.
- Added a video picker, timestamped visual context, exact detector-count answers, and retained-video API endpoints.
- Added per-video follow-up context and clarified the two-frame local versus 12-frame Gemini limits.
- Kept the Q&A/search panel selected after submissions, changed video questions to a standard form, and delayed frontend startup until RAG is healthy.
- Improved local follow-ups with user-question context without feeding prior assistant answers back into the model; retained the two-frame local limit to avoid timeouts.
- Added persistent semantic-search history with saved result snapshots and a dedicated panel to reopen the 50 latest searches without rerunning them.

## 2026-11-06

### Added

- Added batch event ingestion to the RAG service and automatic indexing of general-object labels, counts, and sampled timestamps after video analysis.
- Added frontend status messages for whether video findings were saved to searchable memory.

## 2026-11-02

### Added

- Added video upload and preview to the Streamlit interface.
- Added video analysis through `POST /analyze_video` in the vision service.
- Added timestamped general-object detections, per-class counts, and a summary of the analyzed video.
- Added support for AVI, M4V, MKV, MOV, MP4, MPEG, MPG, and WebM uploads, with a 200 MB size limit.
- Added temporary-file cleanup after video analysis and clear errors for unsupported formats or unreadable videos.
- Added this README with Docker startup instructions, service addresses, API usage, and model limitations.

### Fixed

- Corrected frontend-to-backend URLs to use HTTP on the Docker Compose network.
- Pinned NumPy and compatible Transformers/Hugging Face Hub versions so the RAG service can start with the existing ChromaDB, Sentence Transformers, and PyTorch versions.
- Added `python-multipart` for FastAPI multipart video uploads.

### Limitations

- The bundled YOLOv8n COCO model detects general objects; it is not a PPE or safety-violation detector.
- Video-analysis results are not automatically saved to the RAG database, and uploaded files are not retained after analysis.
- Video analysis samples about one frame per second; supported codecs depend on the OpenCV build.
