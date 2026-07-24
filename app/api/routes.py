from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    CandidateRecommendationRequest,
    InternshipRecommendationRequest,
    IngestRequest
)
from app.services.ingest_service import ingest_document
from app.services.recommendation_service import (
    recommend_candidates,
    recommend_internships
)

router = APIRouter()


@router.post("/ingest")
def ingest(
    request: IngestRequest
):
    try:
        return ingest_document(request)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.post("/recommend/candidates")
def recommend_candidate_profiles(
    request: CandidateRecommendationRequest
):
    try:
        return recommend_candidates(request)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.post("/recommend/internships")
def recommend_internship_postings(
    request: InternshipRecommendationRequest
):
    try:
        return recommend_internships(request)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )