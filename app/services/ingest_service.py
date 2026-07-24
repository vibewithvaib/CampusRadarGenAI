from app.models.schemas import IngestRequest
from app.repositories.chroma_repository import save_document


def ingest_document(
    request: IngestRequest
) -> dict:
    save_document(
        document_id=request.document_id,
        document_type=request.document_type,
        content=request.content
    )

    return {
        "status": "success",
        "message": "Document ingested successfully."
    }