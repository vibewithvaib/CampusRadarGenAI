from app.ai.embeddings import embeddings
from app.core.config import settings
from app.db.chroma import collection


def save_document(
    document_id: str,
    document_type: str,
    content: str
):
    embedding = embeddings.embed_query(content)

    collection.upsert(
        ids=[document_id],
        documents=[content],
        embeddings=[embedding],
        metadatas=[
            {
                "document_type": document_type
            }
        ]
    )


def retrieve_documents(
    query: str,
    document_type: str
) -> list[str]:
    query_embedding = embeddings.embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=settings.MAX_RECOMMENDATIONS,
        where={
            "document_type": document_type
        }
    )

    return results["documents"][0] if results["documents"] else []


def delete_document(
    document_id: str
):
    collection.delete(
        ids=[document_id]
    )