import json
import os

import httpx
from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session

from shared.database import db_dependency
from shared.utils import decode_token
from .crud import save_student_vector, load_recent_vectors, upsert_recommendation_result, get_latest_recommendation
from .recommendation_logic import (
    CourseItem,
    StudentVector,
    build_student_feature_vector,
    recommend_with_kmeans_and_cbf,
    normalize_program,
)
from .schemas import RecommendIn, RecommendOut

router = APIRouter()

PROFILE_SERVICE_URL = os.getenv("PROFILE_SERVICE_URL", "https://profileservice-production-profile.up.railway.app", ).rstrip("/")
COURSE_SERVICE_URL = os.getenv("COURSE_SERVICE_URL", "https://course-service-production-csp.up.railway.app", ).rstrip("/")

SERVICE_TOKEN = os.getenv("SERVICE_TOKEN", "")


def build_router(SessionLocal):
    get_db = db_dependency(SessionLocal)
    JWT_SECRET = os.getenv("JWT_SECRET", "")
    JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
    SERVICE_TOKEN = os.getenv("SERVICE_TOKEN", "")

    if not JWT_SECRET:
        raise RuntimeError("JWT_SECRET not configured")
    if not SERVICE_TOKEN:
        raise RuntimeError("SERVICE_TOKEN not configured")

    def current_user_id(
        authorization: str | None = Header(default=None),
        x_user_id: str | None = Header(default=None, alias="X-User-ID"),
    ) -> int:
        if authorization and authorization.lower().startswith("bearer "):
            token = authorization.split(" ", 1)[1].strip()
            try:
                data = decode_token(token, JWT_SECRET, JWT_ALGORITHM)
                sub = data.get("sub")
                if not sub:
                    raise HTTPException(status_code=401, detail="Token missing sub")
                uid = int(sub)
                if uid <= 0:
                    raise HTTPException(status_code=401, detail="Invalid user id")
                return uid
            except Exception:
                raise HTTPException(status_code=401, detail="Invalid token")

        if not x_user_id:
            raise HTTPException(status_code=401, detail="Missing Authorization or X-User-ID")

        try:
            uid = int(x_user_id)
        except ValueError:
            raise HTTPException(status_code=401, detail="Invalid X-User-ID")

        if uid <= 0:
            raise HTTPException(status_code=401, detail="Invalid user id")
        return uid

    def ensure_service_access(x_service_token: str | None) -> None:
        if not SERVICE_TOKEN:
            raise HTTPException(status_code=500, detail="SERVICE_TOKEN not configured")
        if x_service_token != SERVICE_TOKEN:
            raise HTTPException(status_code=403, detail="Forbidden")

    @router.get("/recommendations/")
    def get_recommendation(user_id: int, db: Session = Depends(get_db)):
        result = get_latest_recommendation(db, user_id)

        if not result:
            return {
                "message": "No results yet",
                "course_recommendations": []
            }

        return {
            "user_id": result.user_id,
            "recommended_program": result.program,
            "confidence": result.confidence,
            "percent_score": result.percent_score,
            "gwa": result.gwa,
            "rating": result.rating,
            "gwa_remarks": result.gwa_remarks,
            "message": result.message,
            "preferred_program": result.preferred_program,
            "weighted_scores": json.loads(result.weighted_scores_json or "{}"),
            "profile_scores": json.loads(result.profile_scores_json or "{}"),
            "cluster_id": result.cluster_id,
            "top_programs": json.loads(result.top_programs_json or "[]"),
            "course_recommendations": [],
        }

    @router.post("/recommend", response_model=RecommendOut)
    async def recommend(payload: RecommendIn, db: Session = Depends(get_db)):
        # 1) fetch profile (for CBF + preferred program)
        interests = ""
        career_goals = ""
        strand = ""
        skills = ""
        preferred_program = ""

        try:
            profile_url = f"{PROFILE_SERVICE_URL}/profile/by-user/{payload.user_id}"
            async with httpx.AsyncClient(timeout=10.0) as client:
                prof_r = await client.get(
                    profile_url,
                    headers={"X-Service-Token": SERVICE_TOKEN},
                )

            print("PROFILE URL:", profile_url)
            print("SERVICE TOKEN EXISTS:", bool(SERVICE_TOKEN))
            print("PROFILE STATUS:", prof_r.status_code)
            print("PROFILE BODY:", prof_r.text)

            if prof_r.status_code == 200:
                prof = prof_r.json()
                interests = prof.get("interests", "") or ""
                career_goals = prof.get("career_goals", "") or ""
                strand = prof.get("strand", "") or ""
                skills = prof.get("skills", "") or ""
                preferred_program = normalize_program(prof.get("preferred_program", "") or "")
                print("PREFERRED PROGRAM READ:", preferred_program)
        except Exception as e:
            print("PROFILE FETCH ERROR:", repr(e))

        if skills:
            interests = f"{interests} {skills}".strip()

        # 2) fetch courses (for CBF)
        courses: list[CourseItem] = []
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                courses_r = await client.get(f"{COURSE_SERVICE_URL}/courses/")

            if courses_r.status_code == 200:
                for c in courses_r.json():
                    courses.append(
                        CourseItem(
                            id=int(c["id"]),
                            code=str(c["code"]),
                            title=str(c["title"]),
                            description=str(c.get("description", "")),
                            program=str(c.get("program", "")).strip().upper(),
                            level=str(c.get("level", "")).strip(),
                            tags=str(c.get("tags", "")).strip(),
                        )
                    )
        except Exception:
            pass

        # 3) build and save THIS student's vector
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
            user_skills=list(map(str, skills.split(","))),
            user_interests=list(map(str, interests.split(","))),
            user_career_goals=list(map(str, career_goals.split(","))),
        )

        # 4) load historical vectors for K-Means
        rows = load_recent_vectors(db, limit=500)
        historical_students: list[StudentVector] = []
        for r in rows:
            try:
                feats = json.loads(r.features_json or "[]")
                if isinstance(feats, list) and feats:
                    historical_students.append(
                        StudentVector(
                            user_id=int(r.user_id),
                            features=[float(x) for x in feats],
                        )
                    )
            except Exception:
                continue

        # 5) compute final recommendation
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
            behavior_score=0.0,
            historical_students=historical_students if len(historical_students) >= 10 else None,
            courses=courses if courses else None,
            top_n_courses=10,
        )

        # make sure response always includes normalized preferred program
        result["preferred_program"] = preferred_program

        # =========================
        # ✅ CLEAN TOP PROGRAMS
        # =========================
        scores = result.get("weighted_scores", {}) or {}
        top_programs = sorted(scores, key=lambda k: scores[k], reverse=True)
        

        # 6) upsert result
        upsert_recommendation_result(
            db,
            user_id=payload.user_id,
            attempt_id=payload.attempt_id,
            program=result["recommended_program"],
            confidence=int(result["confidence"]),
            message=str(result["message"]),
            percent_score=float(result["percent_score"]),
            gwa=float(result["gwa"]),
            rating=str(result["rating"]),
            gwa_remarks=str(result["gwa_remarks"]),
            preferred_program=preferred_program,
            weighted_scores=result.get("weighted_scores"),
            profile_scores=result.get("profile_scores"),
            cluster_id=result.get("cluster_id", 0),
            top_programs=top_programs,
        )

        return result

    return router