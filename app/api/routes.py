from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    IngestRequest,
    CandidateRecommendationRequest,
    InternshipRecommendationRequest
)

from app.services.ingest_service import ingest_document
from app.services.recommendation_service import (
    recommend_candidates,
    recommend_internships
)

router = APIRouter()


@router.post("/ingest", status_code=200)
def ingest(request: IngestRequest):
    try:
        return ingest_document(request)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.post("/recommend/candidates", status_code=200)
def recommend_candidates_route(
    request: CandidateRecommendationRequest
):
    try:
        return recommend_candidates(request)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.post("/recommend/internships", status_code=200)
def recommend_internships_route(
    request: InternshipRecommendationRequest
):
    try:
        return recommend_internships(request)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )