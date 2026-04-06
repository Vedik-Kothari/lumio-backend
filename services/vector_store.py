import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid

# Load local embedding model
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Support Qdrant Cloud or fallback to local
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if QDRANT_URL and QDRANT_API_KEY:
    print("Connecting to Qdrant Cloud...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    print("Using Local Qdrant Storage. (Add QDRANT_URL and QDRANT_API_KEY to switch to Cloud)")
    QDRANT_PATH = os.path.join(os.getcwd(), "qdrant_data")
    LOCK_FILE = os.path.join(QDRANT_PATH, ".lock")
    if os.path.exists(LOCK_FILE):
        try:
            os.remove(LOCK_FILE)
        except Exception as e:
            print(f"Failed to remove lock file: {e}")

    client = QdrantClient(path=QDRANT_PATH)

COLLECTION_NAME = "multilingual_video_chunks"

def init_qdrant():
    try:
        client.get_collection(COLLECTION_NAME)
    except Exception:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

init_qdrant()

class VectorStore:
    @staticmethod
    def add_chunks(chunks):
        points = []
        for chunk in chunks:
            vector = embedding_model.encode(chunk["text"]).tolist()
            point_id = uuid.uuid4()  # Must be a UUID object, not a string
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "text": chunk["text"],
                        "timestamp": chunk["timestamp"],
                        "frame_path": chunk.get("frame_path", ""),
                        "video_id": chunk["video_id"]
                    }
                )
            )
            
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        return len(points)
    
    @staticmethod
    def search(query: str, limit: int = 4):
        vector = embedding_model.encode(query).tolist()
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=limit
        ).points
        return [res.payload for res in results]
