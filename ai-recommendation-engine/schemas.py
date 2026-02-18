from pydantic import BaseModel, Field
from typing import List, Optional


class RecommendIn(BaseModel):
    user_id: int
    attempt_id: int
    score: int = Field(ge=0)
    total: int = Field(gt=0)

    # optional: category breakdown (future-proof)
    logic: int = 0
    programming: int = 0
    networking: int = 0
    design: int = 0


class CourseRecommendationOut(BaseModel):
    course_id: int
    code: str
    title: str
    program: str
    score: float


class RecommendOut(BaseModel):
    user_id: int
    cluster_id: int = 0

    # ✅ rating outputs
    percent_score: float = Field(ge=0, le=100)
    gwa: float
    rating: str
    gwa_remarks: str

    # ✅ recommendation outputs
    recommended_program: str
    confidence: int = Field(ge=0, le=100)
    message: str

    # ✅ optional course recs
    course_recommendations: List[CourseRecommendationOut] = []