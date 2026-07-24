from app.graph.recommendation_graph import recommendation_graph
from app.models.schemas import (
    CandidateRecommendationRequest,
    InternshipRecommendationRequest
)


def recommend_candidates(
    request: CandidateRecommendationRequest
) -> dict:
    result = recommendation_graph.invoke(
        {
            "source_profile": request.internship_description,
            "target_document_type": "candidate",
            "retrieved_profiles": [],
            "ranked_profiles": [],
            "recommendations": [],
            "reflection": {},
            "retry": False
        }
    )

    return {
        "status": "success",
        "recommendations": result["recommendations"]
    }


def recommend_internships(
    request: InternshipRecommendationRequest
) -> dict:
    result = recommendation_graph.invoke(
        {
            "source_profile": request.candidate_profile,
            "target_document_type": "internship",
            "retrieved_profiles": [],
            "ranked_profiles": [],
            "recommendations": [],
            "reflection": {},
            "retry": False
        }
    )

    return {
        "status": "success",
        "recommendations": result["recommendations"]
    }