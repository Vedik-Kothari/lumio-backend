# Lumio Backend

The Lumio backend is a FastAPI service that handles ingestion, transcription, frame analysis, chunking, vector storage, search, and workbench-style AI outputs.

## Responsibilities

- Receive uploaded videos
- Download videos from YouTube or direct URLs
- Track progressive processing state
- Extract transcript and frame context
- Build multimodal chunks for retrieval
- Store and query vectors in Qdrant
- Generate grounded search and workbench responses
- Serve processed videos and frame snapshots

## Backend Features

- Progressive processing with `/api/progress/{video_id}`
- Transcript-first readiness before full enrichment completes
- Visual deduplication to avoid wasting time on near-identical frames
- Video-scoped and cross-video retrieval
- Workbench modes for higher-level tasks
- Metadata storage for library browsing

## Tech Stack

- FastAPI
- Uvicorn
- Groq APIs
- Sentence Transformers
- Qdrant
- `yt-dlp`
- `ffmpeg`
- Pillow

## Project Files

- `main.py`
  FastAPI entrypoint and API routes
- `services/video_processor.py`
  ingestion pipeline, extraction, chunking, and indexing
- `services/ai_pipeline.py`
  search, generation, and workbench logic
- `services/vector_store.py`
  vector storage and retrieval
- `Dockerfile`
  containerized deployment
- `railway.toml`
  Railway deployment settings

## Environment Variables

Create `backend/.env`:

```env
GROQ_API_KEY=your_groq_api_key
QDRANT_URL=https://your-qdrant-cluster-url
QDRANT_API_KEY=your_qdrant_api_key
```

## Run Locally

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Open:

- API root: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Main API Routes

- `POST /api/upload`
  Upload a local file
- `POST /api/upload-link`
  Ingest a YouTube or direct video link
- `GET /api/progress/{video_id}`
  Poll processing progress
- `POST /api/search`
  Ask grounded search questions
- `POST /api/workbench`
  Run higher-level AI analysis modes
- `GET /api/library`
  List indexed videos and metadata
- `GET /api/video/{video_id}`
  Access processed video playback

## Backend Workflow

1. Save or download the video
2. Extract audio and frames
3. Transcribe speech
4. Caption distinct visual states
5. Chunk transcript and visuals
6. Embed and store in Qdrant
7. Return search-ready status
8. Continue enrichment where applicable

## Deployment Notes

This backend is suitable for:

- Railway
- Render
- Fly.io
- Docker-based platforms

Runtime requirements:

- Python 3.11+
- `ffmpeg` installed and available
- network access to Groq and Qdrant

## Frontend Integration

The frontend should point to this backend using:

```env
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

## Do Not Commit

- `.env`
- `uploads/`
- `frames/`
- `metadata/`
- `qdrant_data/`
- local logs
