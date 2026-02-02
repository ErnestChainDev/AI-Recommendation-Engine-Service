import json
from sqlalchemy.orm import Session
from .models import StudentFeatureVector

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
    row = StudentFeatureVector(
        user_id=user_id,
        attempt_id=attempt_id,
        features_json=json.dumps(features),
        score=score,
        total=total,
        logic=logic,
        programming=programming,
        networking=networking,
        design=design,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row

def load_recent_vectors(db: Session, limit: int = 500) -> list[StudentFeatureVector]:
    return (
        db.query(StudentFeatureVector)
        .order_by(StudentFeatureVector.id.desc())
        .limit(limit)
        .all()
    )
