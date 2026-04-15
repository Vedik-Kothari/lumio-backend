import os
import json
import re
from dotenv import load_dotenv
load_dotenv()  # Loads GROQ_API_KEY from backend/.env automatically

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import shutil
import uuid
from datetime import datetime
from urllib.parse import parse_qs, urlparse, urlencode
from urllib.request import urlopen

from services.video_processor import build_chunks, process_video_pipeline
from services.ai_pipeline import AILogic
from services.runtime_paths import (
    DATA_ROOT,
    UPLOADS_DIR,
    FRAMES_DIR,
    ensure_runtime_dirs,
    metadata_file_path,
    upload_file_path,
    uploaded_matches_pattern,
    video_frames_dir,
)
from services.vector_store import VectorStore

app = FastAPI(title="Multimodal RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ensure_runtime_dirs()

app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")
app.mount("/frames", StaticFiles(directory=FRAMES_DIR), name="frames")

class QueryRequest(BaseModel):
    query: str
    video_id: str | None = None
    video_ids: list[str] | None = None

class WorkbenchRequest(BaseModel):
    mode: str
    video_id: str | None = None
    video_ids: list[str] | None = None
    query: str | None = None
    output_format: str | None = None
    scope: str = "current_video"

class UrlUploadRequest(BaseModel):
    url: str
    video_id: str | None = None

class DeleteVideosRequest(BaseModel):
    video_ids: list[str]

import yt_dlp
from yt_dlp.utils import DownloadError
from youtube_transcript_api import YouTubeTranscriptApi

YTDLP_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
YTDLP_COOKIES_FILE = os.path.join(DATA_ROOT, "yt-dlp-cookies.txt")


def _prepare_ytdlp_cookies_file() -> str | None:
    cookies_path = os.getenv("YTDLP_COOKIES_PATH")
    if cookies_path and os.path.isfile(cookies_path):
        return cookies_path

    cookies_content = os.getenv("YTDLP_COOKIES")
    if cookies_content:
        with open(YTDLP_COOKIES_FILE, "w", encoding="utf-8") as cookie_file:
            cookie_file.write(cookies_content)
        return YTDLP_COOKIES_FILE

    return None


def build_ytdlp_options(video_id: str) -> dict:
    ydl_opts = {
        "outtmpl": os.path.join(UPLOADS_DIR, f"{video_id}.%(ext)s"),
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "retries": 2,
        "http_headers": {
            "User-Agent": os.getenv("YTDLP_USER_AGENT", YTDLP_DEFAULT_USER_AGENT),
        },
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"],
            }
        },
    }

    cookies_file = _prepare_ytdlp_cookies_file()
    if cookies_file:
        ydl_opts["cookiefile"] = cookies_file

    return ydl_opts


def classify_ytdlp_error(error: Exception) -> tuple[int, str]:
    message = str(error)
    lowered = message.lower()

    if "sign in to confirm you’re not a bot" in lowered or "sign in to confirm you're not a bot" in lowered:
        return (
            422,
            "YouTube blocked this download from the cloud server with a bot-check. "
            "Lumio already attempted transcript-first retrieval. If this link still fails, "
            "configure backend env var `YTDLP_COOKIES` or `YTDLP_COOKIES_PATH` with exported YouTube cookies "
            "to enable authenticated fallback downloads.",
        )

    if "http error 429" in lowered or "too many requests" in lowered:
        return (
            429,
            "YouTube rate-limited this server while fetching the video. "
            "Please retry in a bit, or configure yt-dlp cookies for the backend to improve direct-access fallback.",
        )

    if "requested format is not available" in lowered:
        return (
            422,
            "This YouTube video exposed an unusual playback format and the server-side fallback could not select a compatible stream. "
            "Lumio will keep preferring transcript-first retrieval, but this specific link still needs a more permissive fallback path.",
        )

    return 500, message


def is_youtube_url(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return "youtube.com" in host or "youtu.be" in host


def extract_youtube_video_id(url: str) -> str | None:
    parsed = urlparse(url)
    host = parsed.netloc.lower()

    if "youtu.be" in host:
        candidate = parsed.path.strip("/").split("/")[0]
        return candidate or None

    if "youtube.com" in host:
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]
        if parsed.path.startswith(("/shorts/", "/embed/", "/live/")):
            parts = [part for part in parsed.path.split("/") if part]
            return parts[1] if len(parts) > 1 else None

    match = re.search(r"(?:v=|/)([0-9A-Za-z_-]{11})(?:[?&/]|$)", url)
    return match.group(1) if match else None


def fetch_youtube_oembed_metadata(url: str) -> dict:
    endpoint = "https://www.youtube.com/oembed?" + urlencode({"url": url, "format": "json"})
    try:
        with urlopen(endpoint, timeout=8) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as error:
        print(f"Could not fetch YouTube oEmbed metadata: {error}")
        return {}


def fetch_youtube_transcript_segments(video_key: str) -> list[dict]:
    preferred_languages = [
        code.strip()
        for code in os.getenv(
            "YOUTUBE_TRANSCRIPT_LANGUAGES",
            "en,en-US,en-GB,hi,es,fr,de,pt,ja,ko",
        ).split(",")
        if code.strip()
    ]

    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_key)

    transcript = None
    if preferred_languages:
        try:
            transcript = transcript_list.find_transcript(preferred_languages)
        except Exception:
            transcript = None

        if transcript is None:
            try:
                transcript = transcript_list.find_generated_transcript(preferred_languages)
            except Exception:
                transcript = None

    if transcript is None:
        transcript = next(iter(transcript_list), None)

    if transcript is None:
        raise RuntimeError("No transcript could be found for this YouTube video.")

    fetched = transcript.fetch()
    segments = []
    for item in fetched:
        start = float(item.get("start", 0))
        duration = float(item.get("duration", 0))
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        segments.append(
            {
                "start": start,
                "end": start + duration,
                "text": text,
            }
        )
    return segments


async def process_youtube_transcript_pipeline(url: str, video_id: str, progress_callback):
    video_key = extract_youtube_video_id(url)
    if not video_key:
        raise HTTPException(status_code=400, detail="Could not parse the YouTube video ID from that link.")

    progress_callback("Reading YouTube transcript...", 12, phase="transcript_fetch")
    segments = fetch_youtube_transcript_segments(video_key)
    if not segments:
        raise RuntimeError("No transcript lines were returned for this YouTube video.")

    progress_callback("Collecting video metadata...", 28, phase="metadata_fetch")
    oembed = fetch_youtube_oembed_metadata(url)
    duration_seconds = int(segments[-1]["end"]) if segments else None

    write_video_metadata(
        video_id,
        {
            "title": oembed.get("title") or f"YouTube video {video_key}",
            "source_type": "youtube_link",
            "source_url": url,
            "duration_seconds": duration_seconds,
            "channel": oembed.get("author_name"),
            "thumbnail": f"https://i.ytimg.com/vi/{video_key}/hqdefault.jpg",
            "created_at": datetime.utcnow().isoformat() + "Z",
        },
    )

    progress_callback("Building transcript-first search index...", 56, phase="initial_index")
    transcript_chunks = build_chunks(
        video_id=video_id,
        segments=segments,
        frame_captions=[],
        stage="transcript_only",
    )

    if not transcript_chunks:
        raise RuntimeError("Transcript was found, but no searchable chunks could be created.")

    VectorStore.replace_video_chunks(video_id, transcript_chunks)
    progress_callback(
        "Search ready from YouTube transcript.",
        100,
        phase="complete",
        is_search_ready=True,
        is_complete=True,
        search_quality="transcript_only",
        warning="This answer path is transcript-grounded. Deep visual enrichment was skipped for this link.",
    )
    return {
        "chunks": transcript_chunks,
        "search_ready": True,
        "processing_complete": True,
        "search_quality": "transcript_only",
    }

def cleanup_old_files(current_video_id: str):
    try:
        if os.path.exists(UPLOADS_DIR):
            for file_name in os.listdir(UPLOADS_DIR):
                if current_video_id not in file_name:
                    file_path = os.path.join(UPLOADS_DIR, file_name)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
        if os.path.exists(FRAMES_DIR):
            for item in os.listdir(FRAMES_DIR):
                if current_video_id not in item:
                    item_path = os.path.join(FRAMES_DIR, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    elif os.path.isfile(item_path):
                        os.remove(item_path)
    except Exception as e:
        print(f"Cleanup error: {e}")

def delete_video_assets(video_id: str):
    try:
        matches = glob.glob(uploaded_matches_pattern(video_id))
        for file_path in matches:
            if os.path.isfile(file_path):
                os.remove(file_path)

        frame_dir = video_frames_dir(video_id)
        if os.path.isdir(frame_dir):
            shutil.rmtree(frame_dir)

        meta_file = metadata_path(video_id)
        if os.path.isfile(meta_file):
            os.remove(meta_file)

        progress_store.pop(video_id, None)
    except Exception as e:
        print(f"Delete asset error for {video_id}: {e}")

progress_store = {}

def metadata_path(video_id: str) -> str:
    return metadata_file_path(video_id)

def read_video_metadata(video_id: str) -> dict:
    try:
        with open(metadata_path(video_id), "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {"video_id": video_id, "title": video_id}

def write_video_metadata(video_id: str, metadata: dict):
    existing = read_video_metadata(video_id)
    merged = {
        **existing,
        **metadata,
        "video_id": video_id,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(metadata_path(video_id), "w", encoding="utf-8") as file:
        json.dump(merged, file, ensure_ascii=True, indent=2)

def set_progress(video_id: str, status: str, percent: int, **extra):
    progress_store[video_id] = {
        "status": status,
        "percent": percent,
        "phase": extra.pop("phase", "pending"),
        "is_search_ready": extra.pop("is_search_ready", False),
        "is_complete": extra.pop("is_complete", False),
        "search_quality": extra.pop("search_quality", "pending"),
        **extra,
    }
    metadata_updates = {}
    if "is_complete" in progress_store[video_id]:
        metadata_updates["is_complete"] = progress_store[video_id]["is_complete"]
    if "search_quality" in progress_store[video_id]:
        metadata_updates["search_quality"] = progress_store[video_id]["search_quality"]
    if status:
        metadata_updates["status"] = status
    if metadata_updates:
        write_video_metadata(video_id, metadata_updates)

@app.get("/api/progress/{video_id}")
async def get_progress(video_id: str):
    return progress_store.get(
        video_id,
        {
            "status": "Waiting...",
            "percent": 0,
            "phase": "pending",
            "is_search_ready": False,
            "is_complete": False,
            "search_quality": "pending",
        },
    )

@app.post("/api/upload-link")
async def upload_video_link(request: UrlUploadRequest):
    url = request.url
    if not url:
        raise HTTPException(status_code=400, detail="No URL provided")
        
    video_id = request.video_id or str(uuid.uuid4())
    set_progress(video_id, "Preparing source...", 0, phase="preparing")
    
    try:
        def update_progress(status: str, percent: int, **extra):
            set_progress(video_id, status, percent, **extra)

        cleanup_old_files(video_id)

        if is_youtube_url(url):
            try:
                return {
                    "status": "success",
                    "video_id": video_id,
                    **(await process_youtube_transcript_pipeline(url, video_id, update_progress)),
                }
            except Exception as transcript_error:
                print(f"YouTube transcript-first path failed: {transcript_error}")
                update_progress("Transcript path unavailable. Trying direct video access...", 18, phase="download_fallback")

        ydl_opts = build_ytdlp_options(video_id)

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            file_path = ydl.prepare_filename(info)
            write_video_metadata(
                video_id,
                {
                    "title": info.get("title") or video_id,
                    "source_type": "youtube_link",
                    "source_url": url,
                    "duration_seconds": info.get("duration"),
                    "channel": info.get("channel") or info.get("uploader"),
                    "thumbnail": info.get("thumbnail"),
                    "created_at": datetime.utcnow().isoformat() + "Z",
                },
            )
            
        # Run processing pipeline
        result = await process_video_pipeline(file_path, video_id, progress_callback=update_progress)
        return {"status": "success", "video_id": video_id, **result}
    except DownloadError as e:
        status_code, detail = classify_ytdlp_error(e)
        print(f"Error during link processing: {e}")
        set_progress(
            video_id,
            "Link download blocked",
            0,
            phase="error",
            is_complete=True,
            error=detail,
        )
        raise HTTPException(status_code=status_code, detail=detail)
    except Exception as e:
        print(f"Error during link processing: {e}")
        set_progress(video_id, "Error occurred", 0, phase="error", is_complete=True, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...), video_id: str = None):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    video_id = video_id or str(uuid.uuid4())
    set_progress(video_id, "Saving uploaded file...", 0, phase="uploading")
    
    file_extension = file.filename.split(".")[-1]
    file_path = upload_file_path(video_id, file_extension)
    write_video_metadata(
        video_id,
        {
            "title": os.path.splitext(file.filename)[0],
            "source_type": "uploaded_file",
            "original_filename": file.filename,
            "created_at": datetime.utcnow().isoformat() + "Z",
        },
    )
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        def update_progress(status: str, percent: int, **extra):
            set_progress(video_id, status, percent, **extra)

        # Clean old videos from disk
        cleanup_old_files(video_id)

        # Run processing pipeline
        result = await process_video_pipeline(file_path, video_id, progress_callback=update_progress)
        return {"status": "success", "video_id": video_id, **result}
    except Exception as e:
        print(f"Error during video processing: {e}")
        set_progress(video_id, "Error occurred", 0, phase="error", is_complete=True, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search")
async def search(request: QueryRequest):
    query = request.query
    try:
        results = await AILogic.search_and_generate(query, video_id=request.video_id, video_ids=request.video_ids)
        return results
    except Exception as e:
        print(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/workbench")
async def workbench(request: WorkbenchRequest):
    try:
        result = await AILogic.run_workbench(
            mode=request.mode,
            video_id=request.video_id,
            video_ids=request.video_ids,
            query=request.query,
            output_format=request.output_format,
            scope=request.scope,
        )
        return result
    except Exception as e:
        print(f"Error during workbench task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/library")
async def get_library():
    from services.vector_store import VectorStore
    video_ids = VectorStore.list_video_ids()
    videos = []
    for video_id in video_ids:
        meta = read_video_metadata(video_id)
        videos.append(
            {
                "video_id": video_id,
                "title": meta.get("title") or video_id,
                "source_type": meta.get("source_type", "indexed_video"),
                "status": meta.get("status", "Indexed"),
                "search_quality": meta.get("search_quality", "unknown"),
                "is_complete": meta.get("is_complete", False),
                "channel": meta.get("channel"),
                "duration_seconds": meta.get("duration_seconds"),
                "created_at": meta.get("created_at"),
                "thumbnail": meta.get("thumbnail"),
                "thumbnail_url": meta.get("thumbnail"),
            }
        )
    return {"videos": videos}

@app.delete("/api/library")
async def delete_library_videos(request: DeleteVideosRequest):
    from services.vector_store import VectorStore

    if not request.video_ids:
        raise HTTPException(status_code=400, detail="No video_ids provided")

    deleted = []
    for video_id in request.video_ids:
        try:
            VectorStore.delete_video_chunks(video_id)
            delete_video_assets(video_id)
            deleted.append(video_id)
        except Exception as e:
            print(f"Failed to delete {video_id}: {e}")

    return {"deleted_video_ids": deleted}

from fastapi.responses import FileResponse
import glob

@app.get("/api/video/{video_id}")
async def get_video(video_id: str):
    # Find the video file with the matching ID regardless of extension
    matches = glob.glob(uploaded_matches_pattern(video_id))
    if not matches:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Exclude the extracted mp3
    video_matches = [m for m in matches if not m.endswith(".mp3")]
    if not video_matches:
        raise HTTPException(status_code=404, detail="Video not found")
        
    return FileResponse(video_matches[0])

@app.get("/")
async def healthcheck():
    return {"status": "ok", "service": "Multimodal RAG API"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
