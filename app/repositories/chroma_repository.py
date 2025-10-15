from typing import Any, Dict

from app.ai.openai_client import embeddings
from app.db.chroma import collection


# ==========================================================
# ADD DOCUMENT
# ==========================================================

def add_document(
    text: str,
    metadata: Dict[str, Any]
) -> None:
    """
    Stores a document in ChromaDB.

    Metadata should contain at least:
    {
        "id": ...,
        "type": "internship" | "candidate"
    }
    """

    embedding = embeddings.embed_query(text)

    collection.add(
        ids=[str(metadata["id"])],
        documents=[text],
        embeddings=[embedding],
        metadatas=[metadata]
    )


# ==========================================================
# COSINE SIMILARITY SEARCH
# ==========================================================

def similarity_search(
    query: str,
    document_type: str,
    n_results: int
) -> Dict[str, Any]:
    """
    Performs cosine similarity search on ChromaDB.

    document_type:
        internship
        candidate
    """

    query_embedding = embeddings.embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={
            "type": document_type
        }
    )

    return results


# ==========================================================
# GET DOCUMENT BY ID
# ==========================================================

def get_document(
    document_id: int
) -> Dict[str, Any]:

    return collection.get(
        ids=[str(document_id)]
    )


# ==========================================================
# GET ALL DOCUMENTS
# ==========================================================

def get_all_documents(
    document_type: str
) -> Dict[str, Any]:

    return collection.get(
        where={
            "type": document_type
        }
    )


# ==========================================================
# DELETE DOCUMENT
# ==========================================================

def delete_document(
    document_id: int
) -> None:

    collection.delete(
        ids=[str(document_id)]
    )