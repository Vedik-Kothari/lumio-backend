import os
from dotenv import load_dotenv
load_dotenv()  # Loads GROQ_API_KEY from backend/.env automatically

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import shutil
import uuid

from services.video_processor import process_video_pipeline
from services.ai_pipeline import AILogic

app = FastAPI(title="Multimodal RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("frames", exist_ok=True)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/frames", StaticFiles(directory="frames"), name="frames")

class QueryRequest(BaseModel):
    query: str

class UrlUploadRequest(BaseModel):
    url: str
    video_id: str | None = None

import yt_dlp

def cleanup_old_files(current_video_id: str):
    try:
        if os.path.exists(UPLOAD_DIR):
            for file_name in os.listdir(UPLOAD_DIR):
                if current_video_id not in file_name:
                    file_path = os.path.join(UPLOAD_DIR, file_name)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
        frames_base = "frames"
        if os.path.exists(frames_base):
            for item in os.listdir(frames_base):
                if current_video_id not in item:
                    item_path = os.path.join(frames_base, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    elif os.path.isfile(item_path):
                        os.remove(item_path)
    except Exception as e:
        print(f"Cleanup error: {e}")

progress_store = {}

@app.get("/api/progress/{video_id}")
async def get_progress(video_id: str):
    return progress_store.get(video_id, {"status": "Waiting...", "percent": 0})

@app.post("/api/upload-link")
async def upload_video_link(request: UrlUploadRequest):
    url = request.url
    if not url:
        raise HTTPException(status_code=400, detail="No URL provided")
        
    video_id = request.video_id or str(uuid.uuid4())
    progress_store[video_id] = {"status": "Downloading Video...", "percent": 0}
    
    ydl_opts = {
        'outtmpl': os.path.join(UPLOAD_DIR, f'{video_id}.%(ext)s'),
        'format': 'best',  # simple reliable format
    }
    
    try:
        def update_progress(status: str, percent: int):
            progress_store[video_id] = {"status": status, "percent": percent}

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            file_path = ydl.prepare_filename(info)
            
        # Clean old videos from disk
        cleanup_old_files(video_id)
            
        # Run processing pipeline
        result = await process_video_pipeline(file_path, video_id, progress_callback=update_progress)
        return {"status": "success", "video_id": video_id, "chunks": result}
    except Exception as e:
        print(f"Error during link processing: {e}")
        progress_store[video_id] = {"status": "Error occurred", "percent": 0}
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...), video_id: str = None):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    video_id = video_id or str(uuid.uuid4())
    progress_store[video_id] = {"status": "Saving uploaded file...", "percent": 0}
    
    file_extension = file.filename.split(".")[-1]
    file_path = os.path.join(UPLOAD_DIR, f"{video_id}.{file_extension}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        def update_progress(status: str, percent: int):
            progress_store[video_id] = {"status": status, "percent": percent}

        # Clean old videos from disk
        cleanup_old_files(video_id)

        # Run processing pipeline
        result = await process_video_pipeline(file_path, video_id, progress_callback=update_progress)
        return {"status": "success", "video_id": video_id, "chunks": result}
    except Exception as e:
        print(f"Error during video processing: {e}")
        progress_store[video_id] = {"status": "Error occurred", "percent": 0}
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search")
async def search(request: QueryRequest):
    query = request.query
    try:
        results = await AILogic.search_and_generate(query)
        return results
    except Exception as e:
        print(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from fastapi.responses import FileResponse
import glob

@app.get("/api/video/{video_id}")
async def get_video(video_id: str):
    # Find the video file with the matching ID regardless of extension
    matches = glob.glob(os.path.join(UPLOAD_DIR, f"{video_id}.*"))
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

