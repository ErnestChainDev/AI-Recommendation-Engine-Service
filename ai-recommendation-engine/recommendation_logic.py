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
        self._idf = {t: math.log((n_docs + 1) / (df_t + 1)) + 1.0 for t, df_t in df.items()}

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
        top = scored[:max(1, top_n)]

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
            self.centroids = []
            self._fitted = False
            self._dim = 0
            return

        random.seed(self.seed)

        points = [sv.features for sv in data if sv.features]
        if not points:
            self.centroids = []
            self._fitted = False
            self._dim = 0
            return

        dim = len(points[0])
        points = [p for p in points if len(p) == dim]
        if not points:
            self.centroids = []
            self._fitted = False
            self._dim = 0
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

    def predict(self, features: List[float]) -> int:
        if not self._fitted or not self.centroids:
            return 0
        if not features:
            return 0
        if self._dim and len(features) != self._dim:
            return 0
        return self._nearest_centroid_index(features)

    def _nearest_centroid_index(self, p: List[float]) -> int:
        best_i = 0
        best_d = float("inf")
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
# Glue: Program + CBF + KMeans
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


def recommend_program_from_signals(
    score: int,
    total: int,
    logic: int = 0,
    programming: int = 0,
    networking: int = 0,
    design: int = 0,
    cluster_id: int = 0,
) -> Tuple[str, int, str]:
    pct = (score / max(1, total)) * 100.0

    buckets = {
        "BSIS": logic,
        "BSCS": programming,
        "BSIT": networking,
        "BTVTED": design,
    }

    max_val = max(buckets.values())
    top_programs = [k for k, v in buckets.items() if v == max_val]

    program = top_programs[0]
    if len(top_programs) > 1:
        cluster_bias = {0: "BSIS", 1: "BSCS", 2: "BSIT", 3: "BTVTED"}
        program = cluster_bias.get(cluster_id % 4, top_programs[0])

    confidence = int(min(95, max(55, pct)))

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
            "Your strongest performance appeared in networking and technical infrastructure. "
            "Information Technology is a good match because it focuses on networking, "
            "hardware, systems support, and administration."
        ),
        "BTVTED": (
            "You showed stronger results in design and creative technology-related areas. "
            "BTVTED ICT fits this pattern because it focuses on multimedia, design, "
            "digital tools, and technology-supported learning."
        ),
    }

    rationale = (
        f"{explanations.get(program, 'This program best matches your strongest quiz area.')} "
        f"(Logic={logic}, Programming={programming}, Networking={networking}, "
        f"Design={design}, Overall Score={score}/{total} or {pct:.1f}%)."
    )

    return program, confidence, rationale


# ----------------------------
# GWA + Rating + Explainable Message
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
        remarks = "Malakas ang performance mo overall—very strong foundation."
    elif gwa <= 2.25:
        rating = "Very Good"
        remarks = "Maganda ang performance mo—solid yung understanding mo."
    elif gwa <= 2.75:
        rating = "Good"
        remarks = "Okay ang performance—may strengths ka pero may areas pa to improve."
    elif gwa <= 3.00:
        rating = "Satisfactory (Pass)"
        remarks = "Pasado—pero recommended na mag-focus sa weak areas para mas tumaas."
    else:
        rating = "Needs Improvement"
        remarks = "Need pa ng practice—pero kaya ’to with consistent review at drills."

    return round(gwa, 2), rating, remarks, round(percent, 1)


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
) -> str:
    preferred = normalize_program(preferred_program)
    recommended = normalize_program(recommended_program)

    score_map = {
        "BSIS": int(logic),
        "BSCS": int(programming),
        "BSIT": int(networking),
        "BTVTED": int(design),
    }

    area_map = {
        "BSIS": "logic, systems analysis, and business-oriented problem solving",
        "BSCS": "programming, algorithms, and computational problem-solving",
        "BSIT": "networking, infrastructure, and technical support",
        "BTVTED": "design, multimedia, and technology-based teaching tools",
    }

    recommended_label = program_label(recommended)
    preferred_label = program_label(preferred) if preferred else ""

    recommended_area = area_map.get(recommended, "your strongest current skill area")
    preferred_area = area_map.get(preferred, "your preferred skill area")
    recommended_score = score_map.get(recommended, 0)
    preferred_score_value = score_map.get(preferred, 0)

    if preferred and preferred == recommended:
        return (
            f"Your preferred program is {preferred_label}, and your quiz performance supports that choice. "
            f"You showed stronger ability in {recommended_area}, which closely matches the demands of {recommended_label}. "
            f"Based on your current result of {percent_score:.1f}% and a recommendation confidence of {confidence}%, "
            f"{recommended_label} appears to be a strong fit for both your interest and present performance."
        )

    if preferred:
        if recommended_score > preferred_score_value:
            return (
                f"You selected {preferred_label} as your preferred program, but your quiz performance suggests that "
                f"{recommended_label} may currently be a better fit. Your stronger results appeared in {recommended_area}, "
                f"while your performance related to {preferred_area} was not as strong. This does not mean that "
                f"{preferred_label} is impossible for you, but based on your current quiz strengths, {recommended_label} "
                f"is the better match at this time. Your overall score was {percent_score:.1f}% with a recommendation "
                f"confidence of {confidence}%."
            )

        return (
            f"You selected {preferred_label} as your preferred program, but the overall pattern of your quiz answers points "
            f"more toward {recommended_label}. Even if your interest is still in {preferred_label}, your present strengths "
            f"are more aligned with {recommended_area}. This is why the system recommends {recommended_label} instead of "
            f"{preferred_label}. Your overall score was {percent_score:.1f}% with a recommendation confidence of "
            f"{confidence}%."
        )

    return (
        f"Based on your quiz performance, {recommended_label} is the most suitable program for you at this time. "
        f"Your strongest results appeared in {recommended_area}, which closely aligns with the demands of "
        f"{recommended_label}. Your overall score was {percent_score:.1f}% with a recommendation confidence of "
        f"{confidence}%."
    )


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
    )

    strengths_summary = (
        f"Strength Breakdown: Logic={logic}, Programming={programming}, "
        f"Networking={networking}, Design={design}."
    )

    preferred_text = (
        f"Preferred Program: {program_label(preferred_program)}\n"
        if preferred_program
        else "Preferred Program: Not specified\n"
    )

    return (
        f"📊 Quiz Rating: {rating} (Estimated GWA: {gwa})\n"
        f"Score: {score}/{total} ({pct:.1f}%)\n"
        f"Remarks: {gwa_remarks}\n"
        f"{preferred_text}"
        f"Recommended Program: {program_label(recommended_program)}\n\n"
        f"🎯 Recommendation Insight:\n"
        f"{preference_message}\n\n"
        f"📌 Program Basis:\n"
        f"{program_rationale}\n\n"
        f"{strengths_summary}"
    )


def build_student_query_text(
    interests: str,
    career_goals: str,
    year_level: str,
    strengths: Dict[str, int],
    total: int,
    preferred_program: str = "",
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

    return f"{interests} {career_goals} {year_level} {preferred_tokens} {' '.join(strength_terms)}".strip()


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
    year_level: str = "",
    preferred_program: str = "",
    behavior_score: float = 0.0,
    historical_students: Optional[List[StudentVector]] = None,
    courses: Optional[List[CourseItem]] = None,
    top_n_courses: int = 10,
) -> Dict:
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

    program, confidence, rationale = recommend_program_from_signals(
        score=score,
        total=total,
        logic=logic,
        programming=programming,
        networking=networking,
        design=design,
        cluster_id=cluster_id,
    )

    gwa, rating_label, gwa_remarks, pct = compute_gwa_and_rating(score=score, total=total)

    final_message = build_explainable_message(
        gwa=gwa,
        rating=rating_label,
        gwa_remarks=gwa_remarks,
        preferred_program=preferred_program,
        recommended_program=program,
        confidence=confidence,
        score=score,
        total=total,
        logic=logic,
        programming=programming,
        networking=networking,
        design=design,
        program_rationale=rationale,
    )

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
            year_level=year_level,
            strengths=strengths,
            total=total,
            preferred_program=preferred_program,
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
            program_filter=program,
        )

    return {
        "user_id": user_id,
        "cluster_id": cluster_id,
        "percent_score": pct,
        "gwa": gwa,
        "rating": rating_label,
        "gwa_remarks": gwa_remarks,
        "preferred_program": normalize_program(preferred_program) if preferred_program else "",
        "recommended_program": program,
        "confidence": confidence,
        "message": final_message,
        "course_recommendations": cbf_results,
    }


# backward compatible
def recommend_program(
    score: int,
    total: int,
    logic: int = 0,
    programming: int = 0,
    networking: int = 0,
    design: int = 0,
):
    program, confidence, rationale = recommend_program_from_signals(
        score=score,
        total=total,
        logic=logic,
        programming=programming,
        networking=networking,
        design=design,
        cluster_id=0,
    )
    return program, confidence, rationale