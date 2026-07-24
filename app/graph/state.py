from typing import Any, TypedDict


class RecommendationState(TypedDict):
    source_profile: str
    target_document_type: str
    retrieved_profiles: list[str]
    ranked_profiles: list[str]
    recommendations: list[dict[str, Any]]
    reflection: dict[str, Any]
    retry: bool