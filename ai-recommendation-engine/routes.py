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
    get_latest_recommendation,
)
from .recommendation_logic import (
    CourseItem,
    StudentVector,
    CBFRecommender,
    build_student_feature_vector,
    build_student_query_text,
    normalize_program,
    recommend_with_kmeans_and_cbf,
)
from .schemas import RecommendIn, RecommendOut

router = APIRouter()

PROFILE_SERVICE_URL = os.getenv("PROFILE_SERVICE_URL", "").rstrip("/")
COURSE_SERVICE_URL  = os.getenv("COURSE_SERVICE_URL",  "").rstrip("/")
SERVICE_TOKEN       = os.getenv("SERVICE_TOKEN", "")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

async def _fetch_courses() -> list[CourseItem]:
    """
    Fetches all courses from the Course Service and returns them as
    a list of CourseItem objects ready for CBF.
    Returns an empty list on any network / parse error.
    """
    if not COURSE_SERVICE_URL:
        return []

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{COURSE_SERVICE_URL}/courses/")

        if r.status_code != 200:
            return []

        items: list[CourseItem] = []
        for c in r.json():
            items.append(
                CourseItem(
                    id          = int(c["id"]),
                    code        = str(c["code"]),
                    title       = str(c["title"]),
                    description = str(c.get("description", "")),
                    program     = str(c.get("program", "")).upper(),
                    level       = str(c.get("level", "")),
                    tags        = str(c.get("tags",  "")),
                )
            )
        return items
    except Exception:
        return []


def _run_cbf(
    *,
    courses: list[CourseItem],
    recommended_program: str,
    interests: str,
    career_goals: str,
    strand: str,
    logic: int,
    programming: int,
    networking: int,
    design: int,
    total: int,
    preferred_program: str,
    user_skills: list[str],
    user_interests: list[str],
    user_career_goals: list[str],
    top_n: int = 10,
) -> list[dict]:
    """
    Runs the CBF recommender against the supplied courses and returns
    the top-N course recommendations for the given student profile.
    Returns an empty list when no courses are available.
    """
    if not courses:
        return []

    student_text = build_student_query_text(
        interests          = interests,
        career_goals       = career_goals,
        strand             = strand,
        strengths          = {
            "logic":       logic,
            "programming": programming,
            "networking":  networking,
            "design":      design,
        },
        total              = total,
        preferred_program  = preferred_program,
        user_skills        = user_skills,
        user_interests_list     = user_interests,
        user_career_goals_list  = user_career_goals,
    )

    normalised = [
        CourseItem(
            id          = c.id,
            code        = c.code,
            title       = c.title,
            description = c.description,
            program     = normalize_program(c.program),
            level       = c.level,
            tags        = c.tags,
        )
        for c in courses
    ]

    cbf = CBFRecommender()
    cbf.fit(normalised)
    return cbf.recommend(
        student_text   = student_text,
        courses        = normalised,
        top_n          = top_n,
        program_filter = recommended_program,
    )


def _to_str_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value if v]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------

def build_router(SessionLocal):
    get_db        = db_dependency(SessionLocal)
    JWT_SECRET    = os.getenv("JWT_SECRET", "")
    JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

    if not JWT_SECRET:
        raise RuntimeError("JWT_SECRET not configured")

    # ── Auth ─────────────────────────────────────────────────────────────────
    def current_user_id(
        authorization: str | None = Header(default=None),
        x_user_id:     str | None = Header(default=None, alias="X-User-ID"),
    ) -> int:
        if authorization and authorization.lower().startswith("bearer "):
            token = authorization.split(" ", 1)[1].strip()
            try:
                data = decode_token(token, JWT_SECRET, JWT_ALGORITHM)
                sub  = data.get("sub")
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

    # ── GET /recommendations ─────────────────────────────────────────────────
    @router.get("/recommendations", response_model=RecommendOut)
    async def get_recommendation(
        user_id: int = Depends(current_user_id),
        db: Session = Depends(get_db),
    ):
        result = get_latest_recommendation(db, user_id)

        if not result:
            raise HTTPException(status_code=404, detail="No results yet")

        recommended_program = result.program          # DB column is "program"
        preferred_program   = result.preferred_program or ""

        # ── Re-fetch profile so CBF has current skill / interest / goal lists ──
        interests    = ""
        career_goals = ""
        strand       = ""
        user_skills: list[str]       = []
        user_interests: list[str]    = []
        user_career_goals: list[str] = []

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(
                    f"{PROFILE_SERVICE_URL}/profile/by-user/{user_id}",
                    headers={"X-Service-Token": SERVICE_TOKEN},
                )
            if r.status_code == 200:
                prof         = r.json()
                interests    = prof.get("interests",    "") or ""
                career_goals = prof.get("career_goals", "") or ""
                strand       = prof.get("strand",       "") or ""
                skills_raw   = prof.get("skills",       "") or ""
                if skills_raw:
                    interests = f"{interests} {skills_raw}".strip()

                # Structured lists — may be stored as JSON strings or plain lists
                def _parse_list_field(v) -> list[str]:
                    if isinstance(v, list):
                        return [str(x) for x in v if x]
                    if isinstance(v, str) and v.strip():
                        try:
                            parsed = json.loads(v)
                            if isinstance(parsed, list):
                                return [str(x) for x in parsed if x]
                        except Exception:
                            pass
                        return [v.strip()]
                    return []

                user_skills       = _parse_list_field(prof.get("user_skills"))
                user_interests    = _parse_list_field(prof.get("user_interests"))
                user_career_goals = _parse_list_field(prof.get("user_career_goals"))
        except Exception:
            pass

        # Reconstruct quiz sub-scores from stored vector if available ──────────
        # (needed to build the correct CBF student query text)
        # Reconstruct quiz sub-scores from stored vector if available
        logic = programming = networking = design = 0
        total = 1

        try:
            from .models import StudentFeatureVector
            vec_row = (
                db.query(StudentFeatureVector)
                .filter(StudentFeatureVector.user_id == user_id)
                .order_by(StudentFeatureVector.id.desc())
                .first()
            )
            if vec_row:
                logic       = vec_row.logic       or 0
                programming = vec_row.programming or 0
                networking  = vec_row.networking  or 0
                design      = vec_row.design      or 0
                total       = vec_row.total       or 1
                # Also recover structured profile lists from JSON columns
                user_skills       = vec_row.get_user_skills()       or user_skills
                user_interests    = vec_row.get_user_interests()    or user_interests
                user_career_goals = vec_row.get_user_career_goals() or user_career_goals
        except Exception:
            pass

        # ── Re-fetch courses and run CBF ──────────────────────────────────────
        courses        = await _fetch_courses()
        course_recs    = _run_cbf(
            courses            = courses,
            recommended_program = recommended_program,
            interests          = interests,
            career_goals       = career_goals,
            strand             = strand,
            logic              = logic,
            programming        = programming,
            networking         = networking,
            design             = design,
            total              = total,
            preferred_program  = preferred_program,
            user_skills        = user_skills,
            user_interests     = user_interests,
            user_career_goals  = user_career_goals,
        )

        return {
            "user_id":               result.user_id,
            "cluster_id":            result.cluster_id,
            "percent_score":         result.percent_score,
            "gwa":                   result.gwa,
            "rating":                result.rating,
            "gwa_remarks":           result.gwa_remarks,
            "preferred_program":     preferred_program,
            "recommended_program":   recommended_program,
            "confidence":            result.confidence,
            "message":               result.message,
            "ai_explanation":        result.ai_explanation,
            "weighted_scores":       result.get_weighted_scores() or {},
            "profile_scores":        result.get_profile_scores()  or {},
            "decision_basis":        result.decision_basis,
            "top_programs":          result.get_top_programs()    or [],
            "course_recommendations": course_recs,   # ✅ live CBF, not []
        }

    # ── POST /recommend ───────────────────────────────────────────────────────
    @router.post("/recommend", response_model=RecommendOut)
    async def recommend(payload: RecommendIn, db: Session = Depends(get_db)):

        # 1. FETCH PROFILE ────────────────────────────────────────────────────
        interests         = ""
        career_goals      = ""
        strand            = ""
        preferred_program = ""

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(
                    f"{PROFILE_SERVICE_URL}/profile/by-user/{payload.user_id}",
                    headers={"X-Service-Token": SERVICE_TOKEN},
                )
            if r.status_code == 200:
                prof              = r.json()
                interests         = prof.get("interests",         "") or ""
                career_goals      = prof.get("career_goals",      "") or ""
                strand            = prof.get("strand",            "") or ""
                skills_raw        = prof.get("skills",            "") or ""
                preferred_program = normalize_program(
                    prof.get("preferred_program", "") or ""
                )
                if skills_raw:
                    interests = f"{interests} {skills_raw}".strip()
        except Exception:
            pass

        # 2. FETCH COURSES ────────────────────────────────────────────────────
        courses = await _fetch_courses()

        # 3. NORMALISE PAYLOAD LISTS ──────────────────────────────────────────
        payload_skills       = _to_str_list(getattr(payload, "user_skills",       None))
        payload_interests    = _to_str_list(getattr(payload, "user_interests",    None))
        payload_career_goals = _to_str_list(getattr(payload, "user_career_goals", None))

        # 4. SAVE FEATURE VECTOR ──────────────────────────────────────────────
        feature_vec = build_student_feature_vector(
            score          = payload.score,
            total          = payload.total,
            logic          = payload.logic,
            programming    = payload.programming,
            networking     = payload.networking,
            design         = payload.design,
            interests_text = interests,
        )

        save_student_vector(
            db,
            user_id           = payload.user_id,
            attempt_id        = payload.attempt_id,
            features          = feature_vec,
            score             = payload.score,
            total             = payload.total,
            logic             = payload.logic,
            programming       = payload.programming,
            networking        = payload.networking,
            design            = payload.design,
            user_skills       = payload_skills,
            user_interests    = payload_interests,
            user_career_goals = payload_career_goals,
        )

        # 5. LOAD HISTORICAL DATA ─────────────────────────────────────────────
        rows = load_recent_vectors(db)
        historical_students: list[StudentVector] = []

        for row in rows:
            try:
                feats = json.loads(row.features_json or "[]")
                if feats:
                    historical_students.append(
                        StudentVector(
                            user_id  = row.user_id,
                            features = [float(x) for x in feats],
                        )
                    )
            except Exception:
                continue

        # 6. MAIN LOGIC (WITH AI) ─────────────────────────────────────────────
        result = recommend_with_kmeans_and_cbf(
            user_id             = payload.user_id,
            score               = payload.score,
            total               = payload.total,
            logic               = payload.logic,
            programming         = payload.programming,
            networking          = payload.networking,
            design              = payload.design,
            interests           = interests,
            career_goals        = career_goals,
            strand              = strand,
            preferred_program   = preferred_program,
            user_skills         = payload_skills,
            user_interests      = payload_interests,
            user_career_goals   = payload_career_goals,
            historical_students = historical_students if len(historical_students) >= 10 else None,
            courses             = courses if courses else None,
            enable_ai_explanation = True,
        )

        # 7. DETERMINE DECISION BASIS ─────────────────────────────────────────
        scores    = result.get("weighted_scores") or {}
        max_score = max(scores.values()) if scores else 0
        top_programs = [p for p, s in scores.items() if s == max_score]

        if len(top_programs) == 1:
            decision_basis = "weighted_score"
        elif preferred_program and preferred_program in top_programs:
            decision_basis = "preferred_program"
        else:
            decision_basis = "tie_breaker"

        # 8. SAVE RESULT ──────────────────────────────────────────────────────
        upsert_recommendation_result(
            db,
            user_id           = payload.user_id,
            attempt_id        = payload.attempt_id,
            program           = result["recommended_program"],
            confidence        = result["confidence"],
            message           = result["message"],
            percent_score     = result["percent_score"],
            gwa               = result["gwa"],
            rating            = result["rating"],
            gwa_remarks       = result["gwa_remarks"],
            preferred_program = preferred_program,
            weighted_scores   = result.get("weighted_scores"),
            profile_scores    = result.get("profile_scores"),
            cluster_id        = result.get("cluster_id", 0),
            top_programs      = top_programs,
            ai_explanation    = result.get("ai_explanation", ""),
            decision_basis    = decision_basis,
        )

        result["decision_basis"] = decision_basis
        result["top_programs"]   = top_programs

        return result

    return router