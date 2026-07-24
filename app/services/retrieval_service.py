from app.repositories.chroma_repository import retrieve_documents


def retrieve_profiles(
    query: str,
    document_type: str
) -> list[str]:
    return retrieve_documents(
        query=query,
        document_type=document_type
    )