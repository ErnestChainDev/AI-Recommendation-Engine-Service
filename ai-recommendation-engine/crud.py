import json
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from .models import StudentFeatureVector, RecommendationResult


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
) -> StudentFeatureVector:
    """
    Upsert:
    - If (user_id, attempt_id) exists -> update it
    - Else -> insert new
    """
    features_json = json.dumps(features)

    existing = (
        db.query(StudentFeatureVector)
        .filter(
            StudentFeatureVector.user_id == user_id,
            StudentFeatureVector.attempt_id == attempt_id,
        )
        .first()
    )

    if existing:
        existing.features_json = features_json
        existing.score = score
        existing.total = total
        existing.logic = logic
        existing.programming = programming
        existing.networking = networking
        existing.design = design

        try:
            db.commit()
        except Exception:
            db.rollback()
            raise

        db.refresh(existing)
        return existing

    row = StudentFeatureVector(
        user_id=user_id,
        attempt_id=attempt_id,
        features_json=features_json,
        score=score,
        total=total,
        logic=logic,
        programming=programming,
        networking=networking,
        design=design,
    )

    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        existing = (
            db.query(StudentFeatureVector)
            .filter(
                StudentFeatureVector.user_id == user_id,
                StudentFeatureVector.attempt_id == attempt_id,
            )
            .first()
        )
        if not existing:
            raise

        existing.features_json = features_json
        existing.score = score
        existing.total = total
        existing.logic = logic
        existing.programming = programming
        existing.networking = networking
        existing.design = design

        try:
            db.commit()
        except Exception:
            db.rollback()
            raise

        db.refresh(existing)
        return existing
    except Exception:
        db.rollback()
        raise

    db.refresh(row)
    return row


def load_recent_vectors(db: Session, limit: int = 500) -> list[StudentFeatureVector]:
    q = db.query(StudentFeatureVector)

    if hasattr(StudentFeatureVector, "created_at"):
        q = q.order_by(StudentFeatureVector.created_at.desc())
    else:
        q = q.order_by(StudentFeatureVector.id.desc())

    return q.limit(limit).all()


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
) -> RecommendationResult:
    """
    Upsert recommendation result by (user_id, attempt_id).
    Stores both:
    - recommendation (program/confidence/message)
    - rating outputs (percent_score/gwa/rating/gwa_remarks)
    """

    existing = (
        db.query(RecommendationResult)
        .filter(
            RecommendationResult.user_id == user_id,
            RecommendationResult.attempt_id == attempt_id,
        )
        .first()
    )

    if existing:
        existing.program = program
        existing.confidence = int(confidence)
        existing.rationale = str(message)

        # new rating columns
        if hasattr(existing, "percent_score"):
            existing.percent_score = float(percent_score)
        if hasattr(existing, "gwa"):
            existing.gwa = float(gwa)
        if hasattr(existing, "rating"):
            existing.rating = str(rating)
        if hasattr(existing, "gwa_remarks"):
            existing.gwa_remarks = str(gwa_remarks)

        try:
            db.commit()
        except Exception:
            db.rollback()
            raise

        db.refresh(existing)
        return existing

    row = RecommendationResult(
        user_id=user_id,
        attempt_id=attempt_id,
        program=program,
        confidence=int(confidence),
        rationale=str(message),
        percent_score=float(percent_score),
        gwa=float(gwa),
        rating=str(rating),
        gwa_remarks=str(gwa_remarks),
    )

    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        # concurrent insert -> update instead
        db.rollback()
        existing = (
            db.query(RecommendationResult)
            .filter(
                RecommendationResult.user_id == user_id,
                RecommendationResult.attempt_id == attempt_id,
            )
            .first()
        )
        if not existing:
            raise

        existing.program = program
        existing.confidence = int(confidence)
        existing.rationale = str(message)
        if hasattr(existing, "percent_score"):
            existing.percent_score = float(percent_score)
        if hasattr(existing, "gwa"):
            existing.gwa = float(gwa)
        if hasattr(existing, "rating"):
            existing.rating = str(rating)
        if hasattr(existing, "gwa_remarks"):
            existing.gwa_remarks = str(gwa_remarks)

        try:
            db.commit()
        except Exception:
            db.rollback()
            raise

        db.refresh(existing)
        return existing
    except Exception:
        db.rollback()
        raise

    db.refresh(row)
    return row