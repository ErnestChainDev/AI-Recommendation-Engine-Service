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

    user_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    attempt_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)

    program: Mapped[str] = mapped_column(String(20), nullable=False)
    confidence: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    message: Mapped[str] = mapped_column(Text, default="", nullable=False)

    percent_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    gwa: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    rating: Mapped[str] = mapped_column(String(32), default="", nullable=False)
    gwa_remarks: Mapped[str] = mapped_column(Text, default="", nullable=False)

    weighted_scores_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    profile_scores_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)

    preferred_program: Mapped[str] = mapped_column(String(20), default="", nullable=False)

    cluster_id: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    top_programs_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    def set_weighted_scores(self, scores: dict) -> None:
        self.weighted_scores_json = json.dumps(scores or {})

    def set_profile_scores(self, scores: dict) -> None:
        self.profile_scores_json = json.dumps(scores or {})


class StudentFeatureVector(Base):
    __tablename__ = "student_feature_vector"

    __table_args__ = (
        UniqueConstraint("user_id", "attempt_id", name="uq_student_vector_user_attempt"),
        Index("ix_student_vector_user_attempt", "user_id", "attempt_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    user_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    attempt_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)

    features_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)

    score: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    logic: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    programming: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    networking: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    design: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    user_skills_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    user_interests_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    user_career_goals_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    def set_features(self, features: list[float]) -> None:
        self.features_json = json.dumps(features or [])

    def set_user_skills(self, skills: list[str]) -> None:
        self.user_skills_json = json.dumps(skills or [])

    def set_user_interests(self, interests: list[str]) -> None:
        self.user_interests_json = json.dumps(interests or [])

    def set_user_career_goals(self, goals: list[str]) -> None:
        self.user_career_goals_json = json.dumps(goals or [])