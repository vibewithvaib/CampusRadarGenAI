from app.core.config import settings
from app.models.schemas import (
    CandidateRecommendationRequest,
    InternshipRecommendationRequest
)
from app.repositories.chroma_repository import similarity_search
from app.services.ranking_service import rank_profiles

def recommend_candidates(
    request: CandidateRecommendationRequest
) -> dict:

    recommended_ids = rank_profiles(
        source_profile=request.internship_description,
        target_profiles=request.applicant_profiles
    )

    return {
        "recommended_student_ids": recommended_ids
    }


def recommend_internships(
    request: InternshipRecommendationRequest
) -> list:
    
    results = similarity_search(
        query=request.student_profile,
        document_type="internship",
        n_results=settings.MAX_RECOMMENDATIONS
    )

    recommendations = []

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    for i in range(len(ids)):
        recommendations.append(
            {
                "internshipId": metadatas[i]["id"],
                "postingText": documents[i]
            }
        )

    return recommendations