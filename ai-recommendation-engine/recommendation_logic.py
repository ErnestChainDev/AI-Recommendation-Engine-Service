import math
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# ----------------------------
# Utilities: text + similarity
# ----------------------------

_TOKEN_RE = re.compile(r"[a-z0-9\-]+")


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def cosine_sim_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    for k, v in a.items():
        dot += v * b.get(k, 0.0)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def l2_distance(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(n)))


# ----------------------------
# Program normalization
# ----------------------------

_PROGRAM_ALIASES: Dict[str, str] = {
    "BSCS": "BSCS",
    "CS": "BSCS",
    "COMPUTER SCIENCE": "BSCS",
    "COMSCI": "BSCS",
    "BSIT": "BSIT",
    "IT": "BSIT",
    "INFORMATION TECHNOLOGY": "BSIT",
    "BSIS": "BSIS",
    "IS": "BSIS",
    "INFORMATION SYSTEMS": "BSIS",
    "BTVTED": "BTVTED",
    "BTVTED-ICT": "BTVTED",
    "ICT": "BTVTED",
    "TVTED": "BTVTED",
}


def normalize_program(p: str) -> str:
    s = (p or "").strip().upper()
    s = re.sub(r"\s+", " ", s)
    return _PROGRAM_ALIASES.get(s, s)


def program_label(program: str) -> str:
    p = normalize_program(program)
    labels = {
        "BSCS": "BSCS (Computer Science)",
        "BSIT": "BSIT (Information Technology)",
        "BSIS": "BSIS (Information Systems)",
        "BTVTED": "BTVTED ICT",
    }
    return labels.get(p, p or "Unknown Program")


# ----------------------------
# Balanced Profile Mapping (5 items each)
# ----------------------------

PROGRAM_MAPPING: Dict[str, Dict[str, List[str]]] = {
    "BSCS": {
        "interests": [
            "algorithms",
            "artificial intelligence",
            "software engineering",
            "data structures",
            "machine learning",
        ],
        "skills": [
            "programming",
            "algorithm design",
            "logical thinking",
            "debugging",
            "mathematical analysis",
        ],
    },
    "BSIT": {
        "interests": [
            "web development",
            "network administration",
            "system integration",
            "cybersecurity",
            "cloud computing",
        ],
        "skills": [
            "web development",
            "network troubleshooting",
            "system administration",
            "hardware setup",
            "cybersecurity basics",
        ],
    },
    "BSIS": {
        "interests": [
            "business process analysis",
            "data analytics",
            "information management",
            "enterprise systems",
            "project management",
        ],
        "skills": [
            "data analysis",
            "documentation",
            "business communication",
            "system planning",
            "critical thinking",
        ],
    },
    "BTVTED": {
        "interests": [
            "technical skills development",
            "teaching",
            "industrial tools",
            "curriculum design",
            "applied technologies",
        ],
        "skills": [
            "technical teaching",
            "hands-on skills",
            "equipment handling",
            "instructional planning",
            "practical problem solving",
        ],
    },
}


# ----------------------------
# Profile Scoring: Skills + Interests + Career Goals
# ----------------------------

def normalize_text_list(items: List[str]) -> List[str]:
    """Lowercase + strip all items."""
    return [i.lower().strip() for i in items if i and i.strip()]


def score_profile_against_mapping(
    user_items: List[str],
    mapping_items: List[str],
) -> float:
    """
    Returns a score between 0.0 and 1.0 representing how many
    of the user's items match the program's mapping list.
    Uses partial/token-level matching for flexibility.
    """
    if not user_items or not mapping_items:
        return 0.0

    user_tokens_list = [set(tokenize(item)) for item in user_items]
    mapping_tokens_list = [set(tokenize(item)) for item in mapping_items]

    matched = 0
    for u_tokens in user_tokens_list:
        for m_tokens in mapping_tokens_list:
            if u_tokens & m_tokens:  # at least one token overlaps
                matched += 1
                break  # count each user item only once

    return matched / len(mapping_items)  # normalize against mapping size (5)


def score_career_goals_against_mapping(
    career_goals: List[str],
    program: str,
) -> float:
    """
    Career goals are matched against both interests and skills
    of the program mapping since goals can overlap either domain.
    """
    mapping = PROGRAM_MAPPING.get(program, {})
    combined = mapping.get("interests", []) + mapping.get("skills", [])
    if not combined:
        return 0.0

    user_tokens_list = [set(tokenize(g)) for g in career_goals if g.strip()]
    matched = 0
    combined_token_sets = [set(tokenize(item)) for item in combined]

    for u_tokens in user_tokens_list:
        for m_tokens in combined_token_sets:
            if u_tokens & m_tokens:
                matched += 1
                break

    return min(1.0, matched / max(1, len(combined_token_sets)))


def compute_profile_scores(
    user_skills: List[str],
    user_interests: List[str],
    user_career_goals: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Returns per-program breakdown:
      { "BSCS": { "skills": 0.8, "interests": 0.6, "career_goals": 0.4 }, ... }
    """
    result: Dict[str, Dict[str, float]] = {}
    norm_skills = normalize_text_list(user_skills)
    norm_interests = normalize_text_list(user_interests)
    norm_goals = normalize_text_list(user_career_goals)

    for program, mapping in PROGRAM_MAPPING.items():
        skills_score = score_profile_against_mapping(norm_skills, mapping.get("skills", []))
        interests_score = score_profile_against_mapping(norm_interests, mapping.get("interests", []))
        career_score = score_career_goals_against_mapping(norm_goals, program)
        result[program] = {
            "skills": round(skills_score, 4),
            "interests": round(interests_score, 4),
            "career_goals": round(career_score, 4),
        }

    return result


# ----------------------------
# Weighted Recommendation Formula
# Recommendation = (Quiz × 60%) + (Skills × 20%) + (Interests × 10%) + (Career Goals × 10%)
# ----------------------------

WEIGHT_QUIZ = 0.60
WEIGHT_SKILLS = 0.20
WEIGHT_INTERESTS = 0.10
WEIGHT_CAREER_GOALS = 0.10


def compute_weighted_scores(
    quiz_score: int,
    quiz_total: int,
    logic: int,
    programming: int,
    networking: int,
    design: int,
    profile_scores: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """
    Computes the final weighted score per program.

    Quiz component: uses per-category subscores mapped to each program.
      BSCS  -> programming
      BSIT  -> networking
      BSIS  -> logic
      BTVTED-> design

    Profile components (skills, interests, career_goals) come from profile_scores.
    """
    quiz_total = max(1, quiz_total)
    overall_pct = (quiz_score / quiz_total)  # 0.0 – 1.0

    # Per-program quiz sub-score (normalized to 0–1)
    program_quiz_map = {
        "BSCS": (programming / quiz_total),
        "BSIT": (networking / quiz_total),
        "BSIS": (logic / quiz_total),
        "BTVTED": (design / quiz_total),
    }

    weighted: Dict[str, float] = {}
    for program in PROGRAM_MAPPING:
        # Blend overall quiz pct (50%) + category sub-score (50%) for quiz component
        quiz_component = (overall_pct * 0.5) + (program_quiz_map.get(program, 0.0) * 0.5)

        p_scores = profile_scores.get(program, {})
        skills_component = p_scores.get("skills", 0.0)
        interests_component = p_scores.get("interests", 0.0)
        career_component = p_scores.get("career_goals", 0.0)

        final = (
            (quiz_component * WEIGHT_QUIZ)
            + (skills_component * WEIGHT_SKILLS)
            + (interests_component * WEIGHT_INTERESTS)
            + (career_component * WEIGHT_CAREER_GOALS)
        )
        weighted[program] = round(final, 6)

    return weighted


def pick_recommended_program(weighted_scores: Dict[str, float]) -> str:
    """Returns the program with the highest weighted score."""
    if not weighted_scores:
        return "BSIT"
    return max(weighted_scores, key=lambda p: weighted_scores[p])


def compute_confidence(weighted_scores: Dict[str, float], recommended: str) -> int:
    """
    Confidence = recommended_score / max_possible * 100, clamped 50–97.
    Adjusted by margin over second-best.
    """
    if not weighted_scores:
        return 50

    sorted_scores = sorted(weighted_scores.values(), reverse=True)
    top = sorted_scores[0]
    second = sorted_scores[1] if len(sorted_scores) > 1 else 0.0

    margin = top - second
    raw_conf = int(min(97, max(50, top * 100)))

    # Boost confidence if margin is large
    if margin >= 0.10:
        raw_conf = min(97, raw_conf + 5)
    elif margin <= 0.02:
        raw_conf = max(50, raw_conf - 5)

    return raw_conf


# ----------------------------
# Content-Based Filtering (CBF)
# ----------------------------

@dataclass
class CourseItem:
    id: int
    code: str
    title: str
    description: str
    program: str
    level: str
    tags: str

    def as_text(self) -> str:
        return f"{self.code} {self.title} {self.description} {self.program} {self.level} {self.tags}"


class CBFRecommender:
    def __init__(self):
        self._idf: Dict[str, float] = {}
        self._course_vecs: Dict[int, Dict[str, float]] = {}
        self._fitted = False

    def fit(self, courses: List[CourseItem]) -> None:
        df: Dict[str, int] = {}
        docs_tokens: Dict[int, List[str]] = {}

        for c in courses:
            toks = tokenize(c.as_text())
            docs_tokens[c.id] = toks
            for t in set(toks):
                df[t] = df.get(t, 0) + 1

        n_docs = max(1, len(courses))
        self._idf = {
            t: math.log((n_docs + 1) / (df_t + 1)) + 1.0
            for t, df_t in df.items()
        }

        self._course_vecs = {}
        for c in courses:
            toks = docs_tokens[c.id]
            tf: Dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            vec: Dict[str, float] = {}
            for t, cnt in tf.items():
                vec[t] = (1.0 + math.log(cnt)) * self._idf.get(t, 0.0)
            self._course_vecs[c.id] = vec

        self._fitted = True

    def _vectorize_query(self, text: str) -> Dict[str, float]:
        toks = tokenize(text)
        tf: Dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        vec: Dict[str, float] = {}
        for t, cnt in tf.items():
            vec[t] = (1.0 + math.log(cnt)) * self._idf.get(t, 0.0)
        return vec

    def recommend(
        self,
        student_text: str,
        courses: List[CourseItem],
        top_n: int = 10,
        program_filter: Optional[str] = None,
    ) -> List[Dict]:
        if not courses:
            return []
        if not self._fitted:
            self.fit(courses)

        qv = self._vectorize_query(student_text)
        pf = normalize_program(program_filter) if program_filter else None

        scored: List[Tuple[int, float]] = []
        for c in courses:
            cp = normalize_program(c.program)
            if pf and cp != pf:
                continue
            cv = self._course_vecs.get(c.id)
            if not cv:
                continue
            s = cosine_sim_sparse(qv, cv)
            scored.append((c.id, s))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[: max(1, top_n)]

        by_id = {c.id: c for c in courses}
        return [
            {
                "course_id": cid,
                "code": by_id[cid].code,
                "title": by_id[cid].title,
                "program": normalize_program(by_id[cid].program),
                "score": round(score, 6),
            }
            for cid, score in top
        ]


# ----------------------------
# K-Means Clustering
# ----------------------------

@dataclass
class StudentVector:
    user_id: int
    features: List[float]


class KMeansClusterer:
    def __init__(self, k: int = 4, max_iter: int = 50, seed: int = 42):
        self.k = k
        self.max_iter = max_iter
        self.seed = seed
        self.centroids: List[List[float]] = []
        self._fitted = False
        self._dim = 0

    def fit(self, data: List[StudentVector]) -> None:
        if not data:
            self._reset()
            return

        random.seed(self.seed)
        points = [sv.features for sv in data if sv.features]
        if not points:
            self._reset()
            return

        dim = len(points[0])
        points = [p for p in points if len(p) == dim]
        if not points:
            self._reset()
            return

        self._dim = dim
        init_k = min(self.k, len(points))
        self.centroids = [p[:] for p in random.sample(points, k=init_k)]
        while len(self.centroids) < self.k:
            self.centroids.append(points[0][:])

        for _ in range(self.max_iter):
            clusters: List[List[List[float]]] = [[] for _ in range(self.k)]
            for p in points:
                idx = self._nearest_centroid_index(p)
                clusters[idx].append(p)

            new_centroids: List[List[float]] = []
            for i in range(self.k):
                if not clusters[i]:
                    new_centroids.append(points[random.randint(0, len(points) - 1)][:])
                else:
                    new_centroids.append(self._mean_vector(clusters[i]))

            shift = sum(l2_distance(a, b) for a, b in zip(self.centroids, new_centroids))
            self.centroids = new_centroids
            if shift < 1e-6:
                break

        self._fitted = True

    def _reset(self):
        self.centroids = []
        self._fitted = False
        self._dim = 0

    def predict(self, features: List[float]) -> int:
        if not self._fitted or not self.centroids or not features:
            return 0
        if self._dim and len(features) != self._dim:
            return 0
        return self._nearest_centroid_index(features)

    def _nearest_centroid_index(self, p: List[float]) -> int:
        best_i, best_d = 0, float("inf")
        for i, c in enumerate(self.centroids):
            d = l2_distance(p, c)
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    @staticmethod
    def _mean_vector(points: List[List[float]]) -> List[float]:
        dim = len(points[0])
        out = [0.0] * dim
        for p in points:
            for j in range(dim):
                out[j] += p[j]
        n = float(len(points))
        return [v / n for v in out]


# ----------------------------
# Feature Vector Builder
# ----------------------------

def build_student_feature_vector(
    score: int,
    total: int,
    logic: int = 0,
    programming: int = 0,
    networking: int = 0,
    design: int = 0,
    interests_text: str = "",
    behavior_score: float = 0.0,
) -> List[float]:
    total = max(1, total)
    overall = (score / total) * 100.0
    logic_pct = (logic / total) * 100.0
    prog_pct = (programming / total) * 100.0
    net_pct = (networking / total) * 100.0
    des_pct = (design / total) * 100.0
    interests_len = float(len(tokenize(interests_text)))
    return [overall, logic_pct, prog_pct, net_pct, des_pct, interests_len, float(behavior_score)]


# ----------------------------
# GWA + Rating
# ----------------------------

def compute_gwa_and_rating(score: int, total: int) -> Tuple[float, str, str, float]:
    total = max(1, total)
    percent = (score / total) * 100.0

    if percent >= 96:
        gwa = 1.00
    elif percent >= 94:
        gwa = 1.25
    elif percent >= 92:
        gwa = 1.50
    elif percent >= 89:
        gwa = 1.75
    elif percent >= 87:
        gwa = 2.00
    elif percent >= 84:
        gwa = 2.25
    elif percent >= 82:
        gwa = 2.50
    elif percent >= 79:
        gwa = 2.75
    elif percent >= 75:
        gwa = 3.00
    else:
        gwa = 5.00

    if gwa <= 1.50:
        rating = "Excellent"
        remarks = "Your overall performance is outstanding, demonstrating a very strong academic foundation."
    elif gwa <= 2.25:
        rating = "Very Good"
        remarks = "Your performance is commendable, showing a solid understanding of the subject matter."
    elif gwa <= 2.75:
        rating = "Good"
        remarks = "Your performance is satisfactory, with evident strengths, though there are areas that require further improvement."
    elif gwa <= 3.00:
        rating = "Satisfactory (Pass)"
        remarks = "You have met the minimum requirements. However, focusing on weaker areas is recommended to improve your overall performance."
    else:
        rating = "Needs Improvement"
        remarks = "Your performance indicates a need for improvement. Consistent practice and review are highly recommended to enhance your understanding."

    return round(gwa, 2), rating, remarks, round(percent, 1)


# ----------------------------
# Explainable Messages
# ----------------------------

def build_preference_aware_program_message(
    *,
    preferred_program: str = "",
    recommended_program: str,
    logic: int,
    programming: int,
    networking: int,
    design: int,
    confidence: int,
    percent_score: float,
    weighted_scores: Optional[Dict[str, float]] = None,
    profile_scores: Optional[Dict[str, Dict[str, float]]] = None,
) -> str:

    recommended = normalize_program(recommended_program)

    # Identify strongest quiz area
    strengths = {
        "BSIS": logic,
        "BSCS": programming,
        "BSIT": networking,
        "BTVTED": design,
    }

    strongest_area_map = {
        "BSCS": "programming",
        "BSIT": "networking",
        "BSIS": "analytical thinking",
        "BTVTED": "design and technical skills",
    }

    strongest_area = strongest_area_map.get(recommended, "your strengths")

    # Profile alignment
    skills_pct = 0
    interests_pct = 0
    goals_pct = 0

    if profile_scores:
        ps = profile_scores.get(recommended, {})
        skills_pct = int(ps.get("skills", 0) * 100)
        interests_pct = int(ps.get("interests", 0) * 100)
        goals_pct = int(ps.get("career_goals", 0) * 100)

    # 🔥 FINAL SHORT EXPLANATION
    return (
        f"You performed strongly in {strongest_area}, and your profile shows "
        f"high alignment in skills ({skills_pct}%), interests ({interests_pct}%), "
        f"and career goals ({goals_pct}%) related to {program_label(recommended)}."
    )


def build_weighted_score_breakdown(
    weighted_scores: Dict[str, float],
    profile_scores: Dict[str, Dict[str, float]],
    recommended: str,
) -> str:
    """Generates a readable breakdown of scores per program."""
    lines = ["📐 Weighted Score Breakdown (Formula: Quiz×60% + Skills×20% + Interests×10% + Goals×10%):"]
    sorted_programs = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)

    for prog, ws in sorted_programs:
        ps = profile_scores.get(prog, {})
        marker = " ✅ Recommended" if normalize_program(prog) == normalize_program(recommended) else ""
        lines.append(
            f"  {program_label(prog)}: {ws * 100:.1f}%"
            f" (Skills={ps.get('skills', 0) * 100:.0f}%,"
            f" Interests={ps.get('interests', 0) * 100:.0f}%,"
            f" Goals={ps.get('career_goals', 0) * 100:.0f}%)"
            f"{marker}"
        )
    return "\n".join(lines)


def build_explainable_message(
    *,
    gwa: float,
    rating: str,
    gwa_remarks: str,
    preferred_program: str = "",
    recommended_program: str,
    confidence: int,
    score: int,
    total: int,
    logic: int,
    programming: int,
    networking: int,
    design: int,
    program_rationale: str,
    weighted_scores: Optional[Dict[str, float]] = None,
    profile_scores: Optional[Dict[str, Dict[str, float]]] = None,
) -> str:
    pct = (score / max(1, total)) * 100.0

    preference_message = build_preference_aware_program_message(
        preferred_program=preferred_program,
        recommended_program=recommended_program,
        logic=logic,
        programming=programming,
        networking=networking,
        design=design,
        confidence=confidence,
        percent_score=pct,
        weighted_scores=weighted_scores,
        profile_scores=profile_scores,
    )

    strengths_summary = (
        f"📊 Quiz Strength Breakdown: Logic={logic}, Programming={programming}, "
        f"Networking={networking}, Design={design}."
    )

    preferred_text = (
        f"Preferred Program  : {program_label(preferred_program)}\n"
        if preferred_program
        else "Preferred Program  : Not specified\n"
    )

    breakdown_text = ""
    if weighted_scores and profile_scores:
        breakdown_text = (
            "\n\n"
            + build_weighted_score_breakdown(weighted_scores, profile_scores, recommended_program)
        )

    return (
        f"📊 Quiz Rating      : {rating} (Estimated GWA: {gwa})\n"
        f"Score              : {score}/{total} ({pct:.1f}%)\n"
        f"Remarks            : {gwa_remarks}\n"
        f"{preferred_text}"
        f"Recommended Program: {program_label(recommended_program)}\n"
        f"\n🎯 Recommendation Insight:\n"
        f"{preference_message}\n"
        f"\n📌 Program Basis:\n"
        f"{program_rationale}\n"
        f"\n{strengths_summary}"
        f"{breakdown_text}"
    )


# ----------------------------
# Student Query Text for CBF
# ----------------------------

def build_student_query_text(
    interests: str,
    career_goals: str,
    strand: str,
    strengths: Dict[str, int],
    total: int,
    preferred_program: str = "",
    user_skills: Optional[List[str]] = None,
    user_interests_list: Optional[List[str]] = None,
    user_career_goals_list: Optional[List[str]] = None,
) -> str:
    total = max(1, total)
    thr = max(1, int(round(total * 0.05)))

    strength_terms: List[str] = []
    if strengths.get("programming", 0) >= thr:
        strength_terms += ["programming", "software", "coding", "algorithms"]
    if strengths.get("networking", 0) >= thr:
        strength_terms += ["networking", "systems", "infrastructure", "security"]
    if strengths.get("logic", 0) >= thr:
        strength_terms += ["analysis", "systems analysis", "requirements", "database"]
    if strengths.get("design", 0) >= thr:
        strength_terms += ["design", "multimedia", "instructional", "teaching"]

    preferred_tokens = ""
    preferred = normalize_program(preferred_program)
    if preferred == "BSCS":
        preferred_tokens = "computer science programming software development algorithms"
    elif preferred == "BSIT":
        preferred_tokens = "information technology networking systems infrastructure support"
    elif preferred == "BSIS":
        preferred_tokens = "information systems analysis database business process"
    elif preferred == "BTVTED":
        preferred_tokens = "btvted ict multimedia design educational technology teaching"

    # Append raw profile lists to enrich the query
    extra = ""
    if user_skills:
        extra += " " + " ".join(normalize_text_list(user_skills))
    if user_interests_list:
        extra += " " + " ".join(normalize_text_list(user_interests_list))
    if user_career_goals_list:
        extra += " " + " ".join(normalize_text_list(user_career_goals_list))

    return (
        f"{interests} {career_goals} {strand} "
        f"{preferred_tokens} {' '.join(strength_terms)} {extra}"
    ).strip()


# ----------------------------
# Program Rationale Builder
# ----------------------------

def build_program_rationale(
    program: str,
    logic: int,
    programming: int,
    networking: int,
    design: int,
    score: int,
    total: int,
    profile_scores: Optional[Dict[str, Dict[str, float]]] = None,
) -> str:
    pct = (score / max(1, total)) * 100.0
    p = normalize_program(program)

    explanations = {
        "BSIS": (
            "You showed stronger logical thinking and analytical skills. "
            "Information Systems fits this pattern because it focuses on logic, "
            "systems analysis, databases, and business processes."
        ),
        "BSCS": (
            "You performed best in programming-related questions. "
            "Computer Science is suitable because it emphasizes coding, algorithms, "
            "problem-solving, and deeper technical development."
        ),
        "BSIT": (
            "Your strongest performance appeared in networking, web development, "
            "and technical infrastructure areas. Information Technology is a good match "
            "because it focuses on networking, web systems, hardware, and administration."
        ),
        "BTVTED": (
            "You showed stronger results in design and creative technology-related areas. "
            "BTVTED ICT fits this pattern because it focuses on multimedia, design, "
            "digital tools, and technology-supported learning."
        ),
    }

    profile_note = ""
    if profile_scores:
        ps = profile_scores.get(p, {})
        skills_pct = int(ps.get("skills", 0) * 100)
        interests_pct = int(ps.get("interests", 0) * 100)
        goals_pct = int(ps.get("career_goals", 0) * 100)
        profile_note = (
            f" Profile alignment — Skills: {skills_pct}%, "
            f"Interests: {interests_pct}%, Career Goals: {goals_pct}%."
        )

    return (
        f"{explanations.get(p, 'This program best matches your strongest area.')} "
        f"(Logic={logic}, Programming={programming}, Networking={networking}, "
        f"Design={design}, Score={score}/{total} [{pct:.1f}%])."
        f"{profile_note}"
    )


# ----------------------------
# Main Entry Point
# ----------------------------

def recommend_with_kmeans_and_cbf(
    *,
    user_id: int,
    score: int,
    total: int,
    logic: int = 0,
    programming: int = 0,
    networking: int = 0,
    design: int = 0,
    interests: str = "",
    career_goals: str = "",
    strand: str = "",
    preferred_program: str = "",
    behavior_score: float = 0.0,
    # NEW: structured profile inputs
    user_skills: Optional[List[str]] = None,
    user_interests: Optional[List[str]] = None,
    user_career_goals: Optional[List[str]] = None,
    historical_students: Optional[List[StudentVector]] = None,
    courses: Optional[List[CourseItem]] = None,
    top_n_courses: int = 10,
) -> Dict:
    """
    Main recommendation function.

    New profile parameters:
      user_skills       — e.g. ["web design", "coding", "cybersecurity basics"]
      user_interests    — e.g. ["web development", "artificial intelligence"]
      user_career_goals — e.g. ["become web developer"]
    """

    # --- 1. K-Means clustering ---
    feature_vec = build_student_feature_vector(
        score=score,
        total=total,
        logic=logic,
        programming=programming,
        networking=networking,
        design=design,
        interests_text=interests,
        behavior_score=behavior_score,
    )

    cluster_id = 0
    if historical_students:
        km = KMeansClusterer(k=4)
        km.fit(historical_students)
        cluster_id = km.predict(feature_vec)

    # --- 2. Profile scoring ---
    _skills = user_skills or []
    _interests = user_interests or []
    _goals = user_career_goals or []

    profile_scores = compute_profile_scores(
        user_skills=_skills,
        user_interests=_interests,
        user_career_goals=_goals,
    )

    # --- 3. Weighted recommendation formula ---
    weighted_scores = compute_weighted_scores(
        quiz_score=score,
        quiz_total=total,
        logic=logic,
        programming=programming,
        networking=networking,
        design=design,
        profile_scores=profile_scores,
    )

    recommended_program = pick_recommended_program(weighted_scores)
    confidence = compute_confidence(weighted_scores, recommended_program)

    # --- 4. GWA + Rating ---
    gwa, rating_label, gwa_remarks, pct = compute_gwa_and_rating(score=score, total=total)

    # --- 5. Rationale ---
    rationale = build_program_rationale(
        program=recommended_program,
        logic=logic,
        programming=programming,
        networking=networking,
        design=design,
        score=score,
        total=total,
        profile_scores=profile_scores,
    )

    # --- 6. Explainable message ---
    final_message = build_explainable_message(
        gwa=gwa,
        rating=rating_label,
        gwa_remarks=gwa_remarks,
        preferred_program=preferred_program,
        recommended_program=recommended_program,
        confidence=confidence,
        score=score,
        total=total,
        logic=logic,
        programming=programming,
        networking=networking,
        design=design,
        program_rationale=rationale,
        weighted_scores=weighted_scores,
        profile_scores=profile_scores,
    )

    # --- 7. CBF course recommendations ---
    cbf_results: List[Dict] = []
    if courses:
        strengths = {
            "logic": logic,
            "programming": programming,
            "networking": networking,
            "design": design,
        }
        student_text = build_student_query_text(
            interests=interests,
            career_goals=career_goals,
            strand=strand,
            strengths=strengths,
            total=total,
            preferred_program=preferred_program,
            user_skills=_skills,
            user_interests_list=_interests,
            user_career_goals_list=_goals,
        )

        normalized_courses = [
            CourseItem(
                id=c.id,
                code=c.code,
                title=c.title,
                description=c.description,
                program=normalize_program(c.program),
                level=c.level,
                tags=c.tags,
            )
            for c in courses
        ]

        cbf = CBFRecommender()
        cbf.fit(normalized_courses)
        cbf_results = cbf.recommend(
            student_text=student_text,
            courses=normalized_courses,
            top_n=top_n_courses,
            program_filter=recommended_program,
        )

    return {
        "user_id": user_id,
        "cluster_id": cluster_id,
        "percent_score": pct,
        "gwa": gwa,
        "rating": rating_label,
        "gwa_remarks": gwa_remarks,
        "preferred_program": normalize_program(preferred_program) if preferred_program else "",
        "recommended_program": recommended_program,
        "confidence": confidence,
        "weighted_scores": weighted_scores,
        "profile_scores": profile_scores,
        "message": final_message,
        "course_recommendations": cbf_results,
    }


# ----------------------------
# Backward-compatible shim
# ----------------------------

def recommend_program(
    score: int,
    total: int,
    logic: int = 0,
    programming: int = 0,
    networking: int = 0,
    design: int = 0,
) -> Tuple[str, int, str]:
    """
    Legacy entry point — no profile inputs.
    Uses only quiz sub-scores to pick a program.
    """
    dummy_profile = compute_profile_scores([], [], [])
    weighted = compute_weighted_scores(
        quiz_score=score,
        quiz_total=total,
        logic=logic,
        programming=programming,
        networking=networking,
        design=design,
        profile_scores=dummy_profile,
    )
    program = pick_recommended_program(weighted)
    confidence = compute_confidence(weighted, program)
    rationale = build_program_rationale(
        program=program,
        logic=logic,
        programming=programming,
        networking=networking,
        design=design,
        score=score,
        total=total,
    )
    return program, confidence, rationale
