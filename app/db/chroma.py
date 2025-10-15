import chromadb

from app.core.config import settings


chroma_client = chromadb.PersistentClient(
    path=settings.CHROMA_DB_PATH
)


collection = chroma_client.get_or_create_collection(
    name=settings.CHROMA_COLLECTION_NAME,
    metadata={
        "hnsw:space": "cosine"
    }
)