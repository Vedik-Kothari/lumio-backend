import os


BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.getenv("APP_DATA_DIR", os.path.join(BACKEND_ROOT, "data"))
UPLOADS_DIR = os.path.join(DATA_ROOT, "uploads")
FRAMES_DIR = os.path.join(DATA_ROOT, "frames")
METADATA_DIR = os.path.join(DATA_ROOT, "metadata")
QDRANT_DIR = os.path.join(DATA_ROOT, "qdrant_data")


def ensure_runtime_dirs():
    for directory in (DATA_ROOT, UPLOADS_DIR, FRAMES_DIR, METADATA_DIR, QDRANT_DIR):
        os.makedirs(directory, exist_ok=True)


def upload_file_path(video_id: str, extension: str) -> str:
    return os.path.join(UPLOADS_DIR, f"{video_id}.{extension}")


def uploaded_matches_pattern(video_id: str) -> str:
    return os.path.join(UPLOADS_DIR, f"{video_id}.*")


def audio_file_path(video_id: str) -> str:
    return os.path.join(UPLOADS_DIR, f"{video_id}.mp3")


def video_frames_dir(video_id: str) -> str:
    return os.path.join(FRAMES_DIR, video_id)


def metadata_file_path(video_id: str) -> str:
    return os.path.join(METADATA_DIR, f"{video_id}.json")


def public_frame_path(video_id: str, file_name: str) -> str:
    return f"frames/{video_id}/{file_name}".replace("\\", "/")
