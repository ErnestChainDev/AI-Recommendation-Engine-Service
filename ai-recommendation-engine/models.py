from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import Integer, String, Text, Float, DateTime, UniqueConstraint, Index, func
from sqlalchemy.orm import Mapped, mapped_column

from shared.database import Base


class RecommendationResult(Base):
    __tablename__ = "recommendation_result"

    __table_args__ = (
        UniqueConstraint("user_id", "attempt_id", name="uq_recommendation_user_attempt"),
        Index("ix_recommendation_user_attempt", "user_id", "attempt_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # 🔹 Basic identifiers
    user_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    attempt_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)

    # 🔹 Recommendation result
    program: Mapped[str] = mapped_column(String(20), nullable=False)
    confidence: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    preferred_program: Mapped[str] = mapped_column(String(20), default="", nullable=False)

    # 🔹 Scores
    percent_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    gwa: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    rating: Mapped[str] = mapped_column(String(32), default="", nullable=False)
    gwa_remarks: Mapped[str] = mapped_column(Text, default="", nullable=False)

    # 🔹 Explainability (RULE-BASED)
    message: Mapped[str] = mapped_column(Text, default="", nullable=False)

    # 🔥 NEW: AI EXPLANATION (XAI)
    ai_explanation: Mapped[str] = mapped_column(Text, default="", nullable=False)

    # 🔥 NEW: reasoning metadata
    decision_basis: Mapped[str] = mapped_column(
        String(50),
        default="weighted_score",
        nullable=False
    )
    # possible values:
    # "weighted_score"
    # "preferred_program"
    # "tie_breaker"

    # 🔹 JSON storage
    weighted_scores_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    profile_scores_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)

    # 🔥 NEW: store tied programs (important for explainability)
    top_programs_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)

    # 🔹 Clustering
    cluster_id: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # 🔹 Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # ----------------------------
    # JSON HELPERS
    # ----------------------------

    def set_weighted_scores(self, scores: dict) -> None:
        self.weighted_scores_json = json.dumps(scores or {})

    def get_weighted_scores(self) -> dict:
        return json.loads(self.weighted_scores_json or "{}")

    def set_profile_scores(self, scores: dict) -> None:
        self.profile_scores_json = json.dumps(scores or {})

    def get_profile_scores(self) -> dict:
        return json.loads(self.profile_scores_json or "{}")

    def set_top_programs(self, programs: list[str]) -> None:
        self.top_programs_json = json.dumps(programs or [])

    def get_top_programs(self) -> list[str]:
        return json.loads(self.top_programs_json or "[]")


# --------------------------------------------------------
# FEATURE VECTOR (NO MAJOR CHANGE - ALREADY GOOD)
# --------------------------------------------------------

class StudentFeatureVector(Base):
    __tablename__ = "student_feature_vector"

    __table_args__ = (
        UniqueConstraint("user_id", "attempt_id", name="uq_student_vector_user_attempt"),
        Index("ix_student_vector_user_attempt", "user_id", "attempt_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    user_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    attempt_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)

    # 🔹 Feature vector for KMeans
    features_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)

    # 🔹 Raw quiz data
    score: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    logic: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    programming: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    networking: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    design: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # 🔹 Structured profile
    user_skills_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    user_interests_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    user_career_goals_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # ----------------------------
    # JSON HELPERS
    # ----------------------------

    def set_features(self, features: list[float]) -> None:
        self.features_json = json.dumps(features or [])

    def get_features(self) -> list[float]:
        return json.loads(self.features_json or "[]")

    def set_user_skills(self, skills: list[str]) -> None:
        self.user_skills_json = json.dumps(skills or [])

    def get_user_skills(self) -> list[str]:
        return json.loads(self.user_skills_json or "[]")

    def set_user_interests(self, interests: list[str]) -> None:
        self.user_interests_json = json.dumps(interests or [])

    def get_user_interests(self) -> list[str]:
        return json.loads(self.user_interests_json or "[]")

    def set_user_career_goals(self, goals: list[str]) -> None:
        self.user_career_goals_json = json.dumps(goals or [])

    def get_user_career_goals(self) -> list[str]:
        return json.loads(self.user_career_goals_json or "[]")