from typing import List, Dict, Optional
from pydantic import BaseModel, Field


# --------------------------------------------------------
# INPUT SCHEMA
# --------------------------------------------------------

class RecommendIn(BaseModel):
    user_id: int
    attempt_id: int

    score: int = Field(ge=0)
    total: int = Field(gt=0)

    logic: int = 0
    programming: int = 0
    networking: int = 0
    design: int = 0

    # 🔥 NEW (profile inputs)
    user_skills: Optional[List[str]] = []
    user_interests: Optional[List[str]] = []
    user_career_goals: Optional[List[str]] = []

    preferred_program: Optional[str] = ""


# --------------------------------------------------------
# COURSE OUTPUT
# --------------------------------------------------------

class CourseRecommendationOut(BaseModel):
    course_id: int
    code: str
    title: str
    program: str
    score: float


# --------------------------------------------------------
# MAIN OUTPUT (XAI READY)
# --------------------------------------------------------

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

    # 🔹 Explainability
    message: str
    ai_explanation: Optional[str] = ""

    # 🔥 NEW (debug / transparency)
    weighted_scores: Dict[str, float] = {}
    profile_scores: Dict[str, Dict[str, float]] = {}

    # 🔥 NEW (decision trace)
    decision_basis: str = "weighted_score"
    top_programs: List[str] = []

    # 🔹 Course recommendations
    course_recommendations: List[CourseRecommendationOut] = []