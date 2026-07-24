from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    document_id: str
    document_type: str = Field(
        description="candidate or internship"
    )
    content: str


class CandidateRecommendationRequest(BaseModel):
    internship_description: str


class InternshipRecommendationRequest(BaseModel):
    candidate_profile: str