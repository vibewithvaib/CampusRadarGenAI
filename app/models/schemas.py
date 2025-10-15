from typing import Any, Dict, List

from pydantic import BaseModel


# ==========================================================
# INGEST DOCUMENT
# ==========================================================

class IngestRequest(BaseModel):
    text: str
    metadata: Dict[str, Any]


# ==========================================================
# RECOMMEND CANDIDATES
# ==========================================================

class CandidateRecommendationRequest(BaseModel):
    internship_description: str
    applicant_profiles: List[str]


# ==========================================================
# RECOMMEND INTERNSHIPS
# ==========================================================

class InternshipRecommendationRequest(BaseModel):
    student_profile: str
    student_skills: List[str] = []