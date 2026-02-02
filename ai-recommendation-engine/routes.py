import json
import os
import httpx
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from shared.database import db_dependency

from .schemas import RecommendIn  # your existing schema
from .models import RecommendationResult
from .crud import save_student_vector, load_recent_vectors
from .recommendation_logic import (
    CourseItem,
    StudentVector,
    build_student_feature_vector,
    recommend_with_kmeans_and_cbf,
)

router = APIRouter()

PROFILE_SERVICE_URL = os.getenv("PROFILE_SERVICE_URL", "http://profile-service:8002")
COURSE_SERVICE_URL = os.getenv("COURSE_SERVICE_URL", "http://course-service:8003")


def build_router(SessionLocal):
    get_db = db_dependency(SessionLocal)

    @router.post("/recommend")
    async def recommend(payload: RecommendIn, db: Session = Depends(get_db)):
        # 1) fetch profile (for CBF)
        interests = ""
        career_goals = ""
        year_level = ""
        skills = ""

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                prof_r = await client.get(f"{PROFILE_SERVICE_URL}/profile/by-user/{payload.user_id}")
            if prof_r.status_code == 200:
                prof = prof_r.json()
                interests = prof.get("interests", "") or ""
                career_goals = prof.get("career_goals", "") or ""
                year_level = prof.get("year_level", "") or ""
                skills = prof.get("skills", "") or ""
        except Exception:
            pass

        if skills:
            interests = f"{interests} {skills}".strip()

        # 2) fetch courses (for CBF)
        courses: list[CourseItem] = []
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                courses_r = await client.get(f"{COURSE_SERVICE_URL}/courses/")
            if courses_r.status_code == 200:
                for c in courses_r.json():
                    courses.append(CourseItem(
                        id=int(c["id"]),
                        code=str(c["code"]),
                        title=str(c["title"]),
                        description=str(c.get("description", "")),
                        program=str(c.get("program", "")),
                        level=str(c.get("level", "")),
                        tags=str(c.get("tags", "")),
                    ))
        except Exception:
            pass

        # 3) build and save THIS student's vector (dataset growth)
        feature_vec = build_student_feature_vector(
            score=payload.score,
            total=payload.total,
            logic=payload.logic,
            programming=payload.programming,
            networking=payload.networking,
            design=payload.design,
            interests_text=interests,
            behavior_score=0.0,
        )

        save_student_vector(
            db,
            user_id=payload.user_id,
            attempt_id=payload.attempt_id,
            features=feature_vec,
            score=payload.score,
            total=payload.total,
            logic=payload.logic,
            programming=payload.programming,
            networking=payload.networking,
            design=payload.design,
        )

        # 4) load historical vectors for K-Means (now real)
        rows = load_recent_vectors(db, limit=500)
        historical_students: list[StudentVector] = []
        for r in rows:
            try:
                feats = json.loads(r.features_json or "[]")
                if isinstance(feats, list) and feats:
                    historical_students.append(StudentVector(user_id=int(r.user_id), features=[float(x) for x in feats]))
            except Exception:
                continue

        # 5) compute final recommendation (program + CBF courses + cluster)
        result = recommend_with_kmeans_and_cbf(
            user_id=payload.user_id,
            score=payload.score,
            total=payload.total,
            logic=payload.logic,
            programming=payload.programming,
            networking=payload.networking,
            design=payload.design,
            interests=interests,
            career_goals=career_goals,
            year_level=year_level,
            behavior_score=0.0,
            historical_students=historical_students if len(historical_students) >= 10 else None,  # avoid garbage clustering early
            courses=courses if courses else None,
            top_n_courses=10,
        )

        # 6) store program recommendation result (optional but useful)
        rec = RecommendationResult(
            user_id=payload.user_id,
            attempt_id=payload.attempt_id,
            program=result["recommended_program"],
            confidence=int(result["confidence"]),
            rationale=str(result["message"]),
        )
        db.add(rec)
        db.commit()

        return result

    return router