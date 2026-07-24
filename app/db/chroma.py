from chromadb import PersistentClient

from app.core.config import settings


client = PersistentClient(
    path=settings.CHROMA_PERSIST_DIRECTORY
)

collection = client.get_or_create_collection(
    name=settings.COLLECTION_NAME
)