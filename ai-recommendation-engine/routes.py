import json
import os

import httpx
from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session

from shared.database import db_dependency
from shared.utils import decode_token
from .crud import (
    save_student_vector,
    load_recent_vectors,
    upsert_recommendation_result,
    get_latest_recommendation
)
from .recommendation_logic import (
    CourseItem,
    StudentVector,
    build_student_feature_vector,
    recommend_with_kmeans_and_cbf,
    normalize_program,
)
from .schemas import RecommendIn, RecommendOut

router = APIRouter()

PROFILE_SERVICE_URL = os.getenv("PROFILE_SERVICE_URL", "").rstrip("/")
COURSE_SERVICE_URL = os.getenv("COURSE_SERVICE_URL", "").rstrip("/")
SERVICE_TOKEN = os.getenv("SERVICE_TOKEN", "")


def build_router(SessionLocal):
    get_db = db_dependency(SessionLocal)
    JWT_SECRET = os.getenv("JWT_SECRET", "")
    JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

    if not JWT_SECRET:
        raise RuntimeError("JWT_SECRET not configured")

    # ----------------------------
    # AUTH
    # ----------------------------
    def current_user_id(
        authorization: str | None = Header(default=None),
        x_user_id: str | None = Header(default=None, alias="X-User-ID"),
    ) -> int:
        if authorization and authorization.lower().startswith("bearer "):
            token = authorization.split(" ", 1)[1].strip()
            try:
                data = decode_token(token, JWT_SECRET, JWT_ALGORITHM)
                sub = data.get("sub")
                if sub is None:
                    raise HTTPException(status_code=401, detail="Invalid token")
                return int(sub)
            except (ValueError, TypeError):
                raise HTTPException(status_code=401, detail="Invalid token")
            except HTTPException:
                raise
            except Exception:
                raise HTTPException(status_code=401, detail="Invalid token")

        if not x_user_id:
            raise HTTPException(status_code=401, detail="Missing auth")

        return int(x_user_id)

    # ----------------------------
    # GET LATEST RESULT
    # ----------------------------
    @router.get("/recommendations", response_model=RecommendOut)
    def get_recommendation(
        user_id: int = Depends(current_user_id),
        db: Session = Depends(get_db),
    ):
        result = get_latest_recommendation(db, user_id)

        if not result:
            raise HTTPException(status_code=404, detail="No results yet")

        return {
            "user_id": result.user_id,
            "cluster_id": result.cluster_id,
            "percent_score": result.percent_score,
            "gwa": result.gwa,
            "rating": result.rating,
            "gwa_remarks": result.gwa_remarks,

            "preferred_program": result.preferred_program,
            "recommended_program": result.program,
            "confidence": result.confidence,

            "message": result.message,
            "ai_explanation": result.ai_explanation,

            "weighted_scores": result.get_weighted_scores(),
            "profile_scores": result.get_profile_scores(),

            "decision_basis": result.decision_basis,
            "top_programs": result.get_top_programs(),

            "course_recommendations": [],
        }

    # ----------------------------
    # MAIN RECOMMENDATION
    # ----------------------------
    @router.post("/recommend", response_model=RecommendOut)
    async def recommend(payload: RecommendIn, db: Session = Depends(get_db)):

        # ----------------------------
        # 1. FETCH PROFILE
        # ----------------------------
        interests = ""
        career_goals = ""
        strand = ""
        skills = ""
        preferred_program = ""

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(
                    f"{PROFILE_SERVICE_URL}/profile/by-user/{payload.user_id}",
                    headers={"X-Service-Token": SERVICE_TOKEN},
                )

            if r.status_code == 200:
                prof = r.json()
                interests = prof.get("interests", "") or ""
                career_goals = prof.get("career_goals", "") or ""
                strand = prof.get("strand", "") or ""
                skills = prof.get("skills", "") or ""
                preferred_program = normalize_program(prof.get("preferred_program", "") or "")

        except Exception:
            pass

        if skills:
            interests = f"{interests} {skills}".strip()

        # ----------------------------
        # 2. FETCH COURSES
        # ----------------------------
        courses: list[CourseItem] = []

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"{COURSE_SERVICE_URL}/courses/")

            if r.status_code == 200:
                for c in r.json():
                    courses.append(
                        CourseItem(
                            id=int(c["id"]),
                            code=str(c["code"]),
                            title=str(c["title"]),
                            description=str(c.get("description", "")),
                            program=str(c.get("program", "")).upper(),
                            level=str(c.get("level", "")),
                            tags=str(c.get("tags", "")),
                        )
                    )
        except Exception:
            pass

        # ----------------------------
        # 3. SAVE FEATURE VECTOR
        # ----------------------------
        feature_vec = build_student_feature_vector(
            score=payload.score,
            total=payload.total,
            logic=payload.logic,
            programming=payload.programming,
            networking=payload.networking,
            design=payload.design,
            interests_text=interests,
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
            user_skills=payload.user_skills,
            user_interests=payload.user_interests,
            user_career_goals=payload.user_career_goals,
        )

        # ----------------------------
        # 4. LOAD HISTORICAL DATA
        # ----------------------------
        rows = load_recent_vectors(db)
        historical_students = []

        for r in rows:
            try:
                feats = json.loads(r.features_json or "[]")
                if feats:
                    historical_students.append(
                        StudentVector(
                            user_id=r.user_id,
                            features=[float(x) for x in feats],
                        )
                    )
            except Exception:
                continue

        # ----------------------------
        # 5. MAIN LOGIC (WITH AI)
        # ----------------------------
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
            strand=strand,
            preferred_program=preferred_program,
            historical_students=historical_students if len(historical_students) >= 10 else None,
            courses=courses if courses else None,
            enable_ai_explanation=True,  # 🔥 important
        )

        # ----------------------------
        # 6. DETERMINE DECISION BASIS
        # ----------------------------
        scores = result.get("weighted_scores", {}) or {}
        max_score = max(scores.values()) if scores else 0

        top_programs = [p for p, s in scores.items() if s == max_score]

        if len(top_programs) == 1:
            decision_basis = "weighted_score"
        elif preferred_program in top_programs:
            decision_basis = "preferred_program"
        else:
            decision_basis = "tie_breaker"

        # ----------------------------
        # 7. SAVE RESULT
        # ----------------------------
        upsert_recommendation_result(
            db,
            user_id=payload.user_id,
            attempt_id=payload.attempt_id,
            program=result["recommended_program"],
            confidence=result["confidence"],
            message=result["message"],
            percent_score=result["percent_score"],
            gwa=result["gwa"],
            rating=result["rating"],
            gwa_remarks=result["gwa_remarks"],
            preferred_program=preferred_program,
            weighted_scores=result.get("weighted_scores"),
            profile_scores=result.get("profile_scores"),
            cluster_id=result.get("cluster_id", 0),
            top_programs=top_programs,
            ai_explanation=result.get("ai_explanation", ""),
            decision_basis=decision_basis,
        )

        # attach missing fields to response
        result["decision_basis"] = decision_basis
        result["top_programs"] = top_programs

        return result

    return router