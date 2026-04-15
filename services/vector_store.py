import hashlib
import os
import threading
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, MatchAny
from sentence_transformers import SentenceTransformer
import uuid
from services.runtime_paths import DATA_ROOT, QDRANT_DIR, ensure_runtime_dirs

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
HF_CACHE_DIR = os.path.join(DATA_ROOT, "huggingface")
SENTENCE_TRANSFORMERS_CACHE_DIR = os.path.join(HF_CACHE_DIR, "sentence_transformers")
os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", SENTENCE_TRANSFORMERS_CACHE_DIR)
_embedding_model = None
_embedding_model_lock = threading.Lock()

# Support Qdrant Cloud or fallback to local
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
ensure_runtime_dirs()
QDRANT_PATH = QDRANT_DIR
LOCK_FILE = os.path.join(QDRANT_PATH, ".lock")

def _build_local_client():
    if os.path.exists(LOCK_FILE):
        try:
            os.remove(LOCK_FILE)
        except Exception as e:
            print(f"Failed to remove lock file: {e}")

    return QdrantClient(path=QDRANT_PATH)

if QDRANT_URL and QDRANT_API_KEY:
    print("Connecting to Qdrant Cloud...")
    primary_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    print("Using Local Qdrant Storage. (Add QDRANT_URL and QDRANT_API_KEY to switch to Cloud)")
    primary_client = _build_local_client()

fallback_client = _build_local_client()

COLLECTION_NAME = "multilingual_video_chunks"

def _iter_clients():
    seen = set()
    for client in (primary_client, fallback_client):
        client_id = id(client)
        if client_id in seen:
            continue
        seen.add(client_id)
        yield client

def _ensure_collection(client) -> bool:
    try:
        client.get_collection(COLLECTION_NAME)
        return True
    except Exception:
        try:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            return True
        except Exception as e:
            print(f"Qdrant collection setup failed: {e}")
            return False

def init_qdrant():
    initialized = False
    for client in _iter_clients():
        initialized = _ensure_collection(client) or initialized

    if not initialized:
        raise RuntimeError("No Qdrant client could be initialized.")

init_qdrant()


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        with _embedding_model_lock:
            if _embedding_model is None:
                print(f"Loading embedding model: {EMBEDDING_MODEL}")
                _embedding_model = SentenceTransformer(
                    EMBEDDING_MODEL,
                    cache_folder=SENTENCE_TRANSFORMERS_CACHE_DIR,
                )
    return _embedding_model

class VectorStore:
    @staticmethod
    def _stage_priority(stage: str) -> int:
        return 0 if stage == "enriched" else 1

    @staticmethod
    def _timestamp_to_seconds(timestamp: str) -> int:
        try:
            start = timestamp.split("-")[0].strip()
            minutes, seconds = start.split(":")
            return int(minutes) * 60 + int(seconds)
        except Exception:
            return 0

    @staticmethod
    def _build_point_id(video_id: str, timestamp: str, text: str) -> str:
        digest = hashlib.sha1(f"{video_id}|{timestamp}|{text}".encode("utf-8")).hexdigest()
        return str(uuid.UUID(digest[:32]))

    @staticmethod
    def add_chunks(chunks):
        embedding_model = get_embedding_model()
        points = []
        for chunk in chunks:
            vector = embedding_model.encode(chunk["text"]).tolist()
            point_id = VectorStore._build_point_id(chunk["video_id"], chunk["timestamp"], chunk["text"])
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "text": chunk["text"],
                        "timestamp": chunk["timestamp"],
                        "frame_path": chunk.get("frame_path", ""),
                        "video_id": chunk["video_id"],
                        "stage": chunk.get("stage", "enriched"),
                    }
                )
            )

        for client in _iter_clients():
            try:
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points
                )
            except Exception as e:
                print(f"Vector upsert failed: {e}")
        return len(points)

    @staticmethod
    def delete_video_chunks(video_id: str):
        deletion_filter = Filter(
            must=[
                FieldCondition(
                    key="video_id",
                    match=MatchValue(value=video_id),
                )
            ]
        )

        for client in _iter_clients():
            try:
                client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=deletion_filter,
                )
            except Exception as e:
                print(f"Vector delete failed: {e}")

    @staticmethod
    def replace_video_chunks(video_id: str, chunks):
        VectorStore.delete_video_chunks(video_id)
        return VectorStore.add_chunks(chunks)
    
    @staticmethod
    def _build_video_filter(video_id: str | None = None, video_ids: list[str] | None = None):
        if video_ids:
            return Filter(
                must=[
                    FieldCondition(
                        key="video_id",
                        match=MatchAny(any=video_ids),
                    )
                ]
            )

        if video_id:
            return Filter(
                must=[
                    FieldCondition(
                        key="video_id",
                        match=MatchValue(value=video_id),
                    )
                ]
            )

        return None

    @staticmethod
    def search(query: str, limit: int = 4, video_id: str | None = None, video_ids: list[str] | None = None):
        embedding_model = get_embedding_model()
        vector = embedding_model.encode(query).tolist()
        query_filter = VectorStore._build_video_filter(video_id=video_id, video_ids=video_ids)

        last_error = None
        for client in (primary_client, fallback_client):
            try:
                results = client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=vector,
                    limit=limit,
                    query_filter=query_filter,
                ).points
                return [res.payload for res in results]
            except Exception as e:
                last_error = e
                print(f"Vector search failed: {e}")

        if last_error:
            raise last_error
        return []

    @staticmethod
    def get_chunks(video_id: str | None = None, video_ids: list[str] | None = None, limit: int = 200):
        scroll_filter = VectorStore._build_video_filter(video_id=video_id, video_ids=video_ids)

        last_error = None
        for client in (primary_client, fallback_client):
            try:
                records, _ = client.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=scroll_filter,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                )
                payloads = [record.payload for record in records if record.payload]
                payloads.sort(
                    key=lambda payload: (
                        payload.get("video_id", ""),
                        VectorStore._stage_priority(payload.get("stage", "")),
                        VectorStore._timestamp_to_seconds(payload.get("timestamp", "")),
                    )
                )
                if video_id or video_ids:
                    deduped = {}
                    for payload in payloads:
                        dedupe_key = f"{payload.get('video_id', '')}:{payload.get('timestamp', '')}"
                        if dedupe_key not in deduped or deduped[dedupe_key].get("stage") != "enriched":
                            deduped[dedupe_key] = payload
                    return sorted(
                        deduped.values(),
                        key=lambda payload: VectorStore._timestamp_to_seconds(payload.get("timestamp", "")),
                    )
                return payloads
            except Exception as e:
                last_error = e
                print(f"Vector scroll failed: {e}")

        if last_error:
            raise last_error
        return []

    @staticmethod
    def list_video_ids(limit: int = 100):
        payloads = VectorStore.get_chunks(limit=limit)
        video_ids = []
        seen = set()
        for payload in payloads:
            video_id = payload.get("video_id")
            if video_id and video_id not in seen:
                seen.add(video_id)
                video_ids.append(video_id)
        return video_ids
