from sqlalchemy import Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from shared.database import Base

class RecommendationResult(Base):
    __tablename__ = "recommendation_result"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True)
    attempt_id: Mapped[int] = mapped_column(Integer, index=True)
    program: Mapped[str] = mapped_column(String(20))
    confidence: Mapped[int] = mapped_column(Integer, default=0)  # 0-100
    rationale: Mapped[str] = mapped_column(Text, default="")

# ✅ NEW: store ML feature vectors for clustering dataset
class StudentFeatureVector(Base):
    __tablename__ = "student_feature_vector"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True)
    attempt_id: Mapped[int] = mapped_column(Integer, index=True)

    # store the numeric vector as JSON string to keep it simple
    features_json: Mapped[str] = mapped_column(Text, default="[]")

    # store breakdown too (helps debugging / analysis)
    score: Mapped[int] = mapped_column(Integer, default=0)
    total: Mapped[int] = mapped_column(Integer, default=0)
    logic: Mapped[int] = mapped_column(Integer, default=0)
    programming: Mapped[int] = mapped_column(Integer, default=0)
    networking: Mapped[int] = mapped_column(Integer, default=0)
    design: Mapped[int] = mapped_column(Integer, default=0)
