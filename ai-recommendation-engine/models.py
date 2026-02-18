from __future__ import annotations

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

    # recommendation outputs
    program: Mapped[str] = mapped_column(String(20), nullable=False)
    confidence: Mapped[int] = mapped_column(Integer, default=0, nullable=False)  # 0-100

    # explainable message (includes GWA + recommendation message)
    rationale: Mapped[str] = mapped_column(Text, default="", nullable=False)

    # rating outputs
    percent_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)  # 0-100
    gwa: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)           # 1.00-5.00
    rating: Mapped[str] = mapped_column(String(32), default="", nullable=False)      # "Excellent", ...
    gwa_remarks: Mapped[str] = mapped_column(Text, default="", nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class StudentFeatureVector(Base):
    __tablename__ = "student_feature_vector"

    __table_args__ = (
        UniqueConstraint("user_id", "attempt_id", name="uq_student_vector_user_attempt"),
        Index("ix_student_vector_user_attempt", "user_id", "attempt_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    user_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    attempt_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)

    # store the numeric vector as JSON string to keep it simple
    features_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)

    # store breakdown too (helps debugging / analysis)
    score: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    logic: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    programming: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    networking: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    design: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )