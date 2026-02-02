import math
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


# ----------------------------
# Utilities: text + similarity
# ----------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")

def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())

def cosine_sim_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
    # cosine between sparse vectors (dicts)
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
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ----------------------------
# Content-Based Filtering (CBF)
# ----------------------------

@dataclass
class CourseItem:
    id: int
    code: str
    title: str
    description: str
    program: str      # "CS" | "IT" | "IS" | "BTVTED"
    level: str
    tags: str         # comma-separated tags

    def as_text(self) -> str:
        # course "content" for CBF
        return f"{self.code} {self.title} {self.description} {self.program} {self.level} {self.tags}"


class CBFRecommender:
    """
    Lightweight TF-IDF + cosine similarity recommender.
    - Fit with course corpus.
    - Build student query text from profile + goals + quiz strengths.
    - Rank courses by similarity.
    """
    def __init__(self):
        self._idf: Dict[str, float] = {}
        self._course_vecs: Dict[int, Dict[str, float]] = {}
        self._fitted = False

    def fit(self, courses: List[CourseItem]) -> None:
        # Compute document frequency
        df: Dict[str, int] = {}
        docs_tokens: Dict[int, List[str]] = {}

        for c in courses:
            toks = tokenize(c.as_text())
            docs_tokens[c.id] = toks
            seen = set(toks)
            for t in seen:
                df[t] = df.get(t, 0) + 1

        n_docs = max(1, len(courses))
        # Smooth IDF
        self._idf = {t: math.log((n_docs + 1) / (df_t + 1)) + 1.0 for t, df_t in df.items()}

        # Build course TF-IDF vectors
        self._course_vecs = {}
        for c in courses:
            toks = docs_tokens[c.id]
            tf: Dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1

            # tf-idf with log-tf
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
        if not self._fitted:
            self.fit(courses)

        qv = self._vectorize_query(student_text)

        scored: List[Tuple[int, float]] = []
        for c in courses:
            if program_filter and c.program != program_filter:
                continue
            cv = self._course_vecs.get(c.id)
            if not cv:
                continue
            s = cosine_sim_sparse(qv, cv)
            scored.append((c.id, s))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_n]

        by_id = {c.id: c for c in courses}
        return [
            {
                "course_id": cid,
                "code": by_id[cid].code,
                "title": by_id[cid].title,
                "program": by_id[cid].program,
                "score": round(score, 6),
            }
            for cid, score in top
        ]


# ----------------------------
# K-Means Clustering (Students)
# ----------------------------

@dataclass
class StudentVector:
    user_id: int
    features: List[float]  # numeric representation


class KMeansClusterer:
    """
    Pure Python K-Means (no numpy/scikit).
    Purpose: group students into similar clusters based on
    performance + interests + behavior signals (as numeric features).

    Usage:
    clusterer = KMeansClusterer(k=4)
    clusterer.fit(student_vectors)
    cluster_id = clusterer.predict(new_student_features)
    """
    def __init__(self, k: int = 4, max_iter: int = 50, seed: int = 42):
        self.k = k
        self.max_iter = max_iter
        self.seed = seed
        self.centroids: List[List[float]] = []
        self._fitted = False

    def fit(self, data: List[StudentVector]) -> None:
        if not data:
            self.centroids = []
            self._fitted = False
            return

        random.seed(self.seed)

        # Initialize centroids by sampling distinct points
        points = [sv.features for sv in data]
        self.centroids = [p[:] for p in random.sample(points, k=min(self.k, len(points)))]

        # If fewer points than k, pad centroids with duplicates (safe)
        while len(self.centroids) < self.k:
            self.centroids.append(points[0][:])

        for _ in range(self.max_iter):
            clusters: List[List[List[float]]] = [[] for _ in range(self.k)]

            # Assign
            for p in points:
                idx = self._nearest_centroid_index(p)
                clusters[idx].append(p)

            # Recompute
            new_centroids = []
            for i in range(self.k):
                if not clusters[i]:
                    # empty cluster -> re-seed from a random point
                    new_centroids.append(points[random.randint(0, len(points) - 1)][:])
                    continue
                new_centroids.append(self._mean_vector(clusters[i]))

            # Check convergence
            shift = sum(l2_distance(a, b) for a, b in zip(self.centroids, new_centroids))
            self.centroids = new_centroids
            if shift < 1e-6:
                break

        self._fitted = True

    def predict(self, features: List[float]) -> int:
        if not self._fitted or not self.centroids:
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
    """
    Numeric vector for K-Means.
    You can expand features later if you store more data.

    Features:
    [ overall_pct, logic_pct, programming_pct, networking_pct, design_pct, interests_len, behavior_score ]
    """
    total = max(1, total)
    overall = (score / total) * 100.0
    logic_pct = (logic / total) * 100.0
    prog_pct = (programming / total) * 100.0
    net_pct = (networking / total) * 100.0
    des_pct = (design / total) * 100.0
    interests_len = float(len(tokenize(interests_text)))

    return [
        overall,
        logic_pct,
        prog_pct,
        net_pct,
        des_pct,
        interests_len,
        float(behavior_score),
    ]


def recommend_program_from_signals(
    score: int,
    total: int,
    logic: int = 0,
    programming: int = 0,
    networking: int = 0,
    design: int = 0,
    cluster_id: int = 0,
) -> Tuple[str, int, str]:
    """
    Program recommendation with human-readable explanation.
    """

    pct = (score / max(1, total)) * 100.0

    # Strength buckets
    buckets = {
        "IS": logic,
        "CS": programming,
        "IT": networking,
        "BTVTED (ICT)": design,
    }

    # Pick strongest area
    program = max(buckets.items(), key=lambda kv: kv[1])[0]

    # Tie-breaker using cluster
    if len(set(buckets.values())) == 1:
        cluster_bias = {
            0: "IS",
            1: "CS",
            2: "IT",
            3: "BTVTED (ICT)",
        }
        program = cluster_bias.get(cluster_id % 4, "IS")

    confidence = int(min(95, max(55, pct)))

    # 🧠 WHY message per program
    explanations = {
        "IS": (
            "You showed strong logical thinking and analytical skills. "
            "Information Systems fits you because it focuses on logic, "
            "systems analysis, and business processes rather than heavy coding."
        ),
        "CS": (
            "You performed best in programming-related questions. "
            "Computer Science is suitable for you because it emphasizes "
            "programming, algorithms, and problem-solving skills."
        ),
        "IT": (
            "Your strength lies in networking and technical infrastructure. "
            "Information Technology matches you well because it focuses on "
            "networking, hardware, and system administration."
        ),
        "BTVTED (ICT)": (
            "You excelled in design and creative tasks. "
            "The BTVTED ICT track is ideal for you because it focuses on "
            "multimedia, design, basic web development, productivity tools, "
            "and teaching with technology."
        ),
    }

    rationale = (
        f"{explanations.get(program)} "
        f"(Logic={logic}, Programming={programming}, "
        f"Networking={networking}, Design={design}, "
        f"Overall Score={score}/{total} or {pct:.1f}%)."
    )

    return program, confidence, rationale



def build_student_query_text(
    interests: str,
    career_goals: str,
    year_level: str,
    strengths: Dict[str, int],
) -> str:
    """
    Builds the 'student content profile' used in CBF matching.
    """
    # Translate strengths into keyword signals that are likely to match course tags/descriptions.
    strength_terms = []
    if strengths.get("programming", 0) > 0:
        strength_terms += ["programming", "software", "coding", "algorithms"]
    if strengths.get("networking", 0) > 0:
        strength_terms += ["networking", "systems", "infrastructure", "security"]
    if strengths.get("logic", 0) > 0:
        strength_terms += ["analysis", "systems analysis", "requirements", "database"]
    if strengths.get("design", 0) > 0:
        strength_terms += ["design", "multimedia", "instructional", "teaching"]

    return f"{interests} {career_goals} {year_level} {' '.join(strength_terms)}"


def recommend_with_kmeans_and_cbf(
    *,
    # student signals
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
    behavior_score: float = 0.0,

    # historical students for clustering (optional, from DB or analytics service)
    historical_students: Optional[List[StudentVector]] = None,

    # course corpus (required for CBF)
    courses: Optional[List[CourseItem]] = None,
    top_n_courses: int = 10,
) -> Dict:
    """
    One-stop function:
    1) K-Means cluster student (if historical_students provided)
    2) Recommend program (CS/IT/IS/BTVTED)
    3) Use CBF to recommend courses aligned with profile + strengths
    """
    # --- KMeans ---
    feature_vec = build_student_feature_vector(
        score=score, total=total,
        logic=logic, programming=programming, networking=networking, design=design,
        interests_text=interests,
        behavior_score=behavior_score,
    )

    cluster_id = 0
    if historical_students:
        km = KMeansClusterer(k=4)
        km.fit(historical_students)
        cluster_id = km.predict(feature_vec)

    # --- Program recommendation ---
    program, confidence, rationale = recommend_program_from_signals(
        score=score, total=total,
        logic=logic, programming=programming, networking=networking, design=design,
        cluster_id=cluster_id,
    )

    # --- CBF course recommendation ---
    cbf_results: List[Dict] = []
    if courses:
        strengths = {"logic": logic, "programming": programming, "networking": networking, "design": design}
        student_text = build_student_query_text(interests, career_goals, year_level, strengths)

        cbf = CBFRecommender()
        cbf.fit(courses)

        # Option A: filter courses by recommended program
        cbf_results = cbf.recommend(
            student_text=student_text,
            courses=courses,
            top_n=top_n_courses,
            program_filter=program,   # remove this if you want cross-program suggestions
        )

    return {
        "user_id": user_id,
        "cluster_id": cluster_id,
        "recommended_program": program,
        "confidence": confidence,
        "message": rationale,
        "course_recommendations": cbf_results,
    }


# ----------------------------
# Backward compatible function
# ----------------------------

def recommend_program(score: int, total: int, logic: int = 0, programming: int = 0, networking: int = 0, design: int = 0):
    """
    Keeps your old signature working.
    Returns: (program, confidence, rationale)
    """
    program, confidence, rationale = recommend_program_from_signals(
        score=score, total=total,
        logic=logic, programming=programming, networking=networking, design=design,
        cluster_id=0,
    )
    return program, confidence, rationale