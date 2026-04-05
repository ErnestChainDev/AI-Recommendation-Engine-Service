from typing import List
from pydantic import BaseModel, Field


class RecommendIn(BaseModel):
    user_id: int
    attempt_id: int
    score: int = Field(ge=0)
    total: int = Field(gt=0)

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

    percent_score: float = Field(ge=0, le=100)
    gwa: float
    rating: str
    gwa_remarks: str

    preferred_program: str = ""
    recommended_program: str
    confidence: int = Field(ge=0, le=100)
    message: str

    course_recommendations: List[CourseRecommendationOut] = []