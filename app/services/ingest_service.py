from app.models.schemas import IngestRequest
from app.repositories.chroma_repository import add_document


def ingest_document(request: IngestRequest) -> dict:
    """
    Stores a candidate or internship document in ChromaDB.
    """

    add_document(
        text=request.text,
        metadata=request.metadata
    )

    return {
        "status": "success"
    }