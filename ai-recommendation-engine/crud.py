import json
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from .models import StudentFeatureVector, RecommendationResult

def load_recent_vectors(db: Session, limit: int = 500):
    return (
        db.query(StudentFeatureVector)
        .order_by(StudentFeatureVector.created_at.desc())
        .limit(limit)
        .all()
    )

def get_latest_recommendation(db: Session, user_id: int):
    return (
        db.query(RecommendationResult)
        .filter(RecommendationResult.user_id == user_id)
        .order_by(RecommendationResult.created_at.desc())
        .first()
    )

def save_student_vector(
    db: Session,
    *,
    user_id: int,
    attempt_id: int,
    features: list[float],
    score: int,
    total: int,
    logic: int,
    programming: int,
    networking: int,
    design: int,
    user_skills: list[str] | None = None,
    user_interests: list[str] | None = None,
    user_career_goals: list[str] | None = None,
) -> StudentFeatureVector:

    def _apply(row: StudentFeatureVector):
        row.set_features(features)
        row.score = score
        row.total = total
        row.logic = logic
        row.programming = programming
        row.networking = networking
        row.design = design

        row.set_user_skills(user_skills or [])
        row.set_user_interests(user_interests or [])
        row.set_user_career_goals(user_career_goals or [])

    existing = db.query(StudentFeatureVector).filter(
        StudentFeatureVector.user_id == user_id,
        StudentFeatureVector.attempt_id == attempt_id
    ).first()

    if existing:
        _apply(existing)
        db.commit()
        db.refresh(existing)
        return existing

    row = StudentFeatureVector(user_id=user_id, attempt_id=attempt_id)
    _apply(row)
    db.add(row)

    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        return save_student_vector(db, **locals())

    db.refresh(row)
    return row


def upsert_recommendation_result(
    db: Session,
    *,
    user_id: int,
    attempt_id: int,
    program: str,
    confidence: int,
    message: str,
    percent_score: float,
    gwa: float,
    rating: str,
    gwa_remarks: str,
    preferred_program: str = "",
    weighted_scores: dict | None = None,
    profile_scores: dict | None = None,
    cluster_id: int = 0,
    top_programs: list[str] | None = None,
) -> RecommendationResult:

    def _apply(row: RecommendationResult):
        row.program = program.upper()
        row.confidence = confidence
        row.message = message

        row.percent_score = percent_score
        row.gwa = gwa
        row.rating = rating
        row.gwa_remarks = gwa_remarks

        row.preferred_program = preferred_program.upper()

        row.set_weighted_scores(weighted_scores or {})
        row.set_profile_scores(profile_scores or {})

        row.cluster_id = cluster_id
        row.top_programs_json = json.dumps(top_programs or [])

    existing = db.query(RecommendationResult).filter(
        RecommendationResult.user_id == user_id,
        RecommendationResult.attempt_id == attempt_id
    ).first()

    if existing:
        _apply(existing)
        db.commit()
        db.refresh(existing)
        return existing

    row = RecommendationResult(user_id=user_id, attempt_id=attempt_id)
    _apply(row)
    db.add(row)

    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        return upsert_recommendation_result(db, **locals())

    db.refresh(row)
    return row