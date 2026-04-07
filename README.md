# Lumio Backend

FastAPI backend for Lumio, an AI video intelligence platform for ingesting videos, indexing transcript plus visual context, and answering grounded questions with timestamped evidence.

## What It Handles

- Uploading local video files
- Importing YouTube and direct video links
- Progressive processing and status reporting
- Transcript-first indexing with later visual enrichment
- Search, workbench outputs, and video library metadata
- Static serving for processed uploads and extracted frame grids

## Stack

- FastAPI
- Uvicorn
- Groq for speech, vision, and answer generation
- Sentence Transformers for embeddings
- Qdrant for vector retrieval
- `yt-dlp` for remote video ingestion
- `ffmpeg` for extraction and processing

## Environment Variables

Create `backend/.env`:

```env
GROQ_API_KEY=your_groq_api_key
QDRANT_URL=https://your-qdrant-cluster-url
QDRANT_API_KEY=your_qdrant_api_key
```

## Local Development

From the `backend` folder:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

API docs will be available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Deployment Notes

This backend is ready to deploy independently to Railway, Render, Fly.io, or any container host.

Required runtime dependencies:

- Python 3.11+
- `ffmpeg` available on the system path
- outbound access to Groq and Qdrant

If you deploy on Railway, the existing `Dockerfile` and `railway.toml` can be used directly.

## Suggested Standalone Repo Contents

When splitting into a backend-only repository, keep:

- `main.py`
- `services/`
- `requirements.txt`
- `Dockerfile`
- `railway.toml`
- `.gitignore`
- `README.md`

Do not commit:

- `.env`
- `uploads/`
- `frames/`
- `metadata/`
- `qdrant_data/`
- crash logs or local dev logs

## Key API Routes

- `POST /api/upload`
- `POST /api/upload-link`
- `GET /api/progress/{video_id}`
- `POST /api/search`
- `POST /api/workbench`
- `GET /api/library`
- `GET /api/video/{video_id}`

## Frontend Integration

The frontend only needs the backend base URL through:

```env
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```
