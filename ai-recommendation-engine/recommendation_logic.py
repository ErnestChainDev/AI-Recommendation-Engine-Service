"""
recommendation_engine.py
========================
Hybrid Academic Program Recommendation System
---------------------------------------------
Combines K-Means Clustering, Content-Based Filtering (CBF),
and a Weighted Scoring Formula to recommend IT-related academic
programs (BSCS, BSIT, BSIS, BTVTED) based on quiz performance
and student profile data.

Explainable AI (XAI) integration via OpenRouter LLM produces
human-readable, advisor-style explanations for each recommendation.

Formula:
    Final Score = (Quiz × 60%) + (Skills × 20%) + (Interests × 10%) + (Career Goals × 10%)

Author  : ErnestChainDev
Version : 4.0.0 (v5 Defense Ready — BERT + Fuzzy + Token Profile Scoring)
"""

from __future__ import annotations

import logging
import math
import os
import random
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# BERT (Sentence Transformers)
# ---------------------------------------------------------------------------

st_util = None

try:
    from sentence_transformers import SentenceTransformer, util as st_util
    BERT_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    _BERT_AVAILABLE = True
except ImportError:
    BERT_MODEL = None
    _BERT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEIGHT_QUIZ: float        = 0.60
WEIGHT_SKILLS: float      = 0.20
WEIGHT_INTERESTS: float   = 0.10
WEIGHT_CAREER_GOALS: float = 0.10

_TOKEN_RE = re.compile(r"[a-z0-9\-]+")

# ---------------------------------------------------------------------------
# Program Metadata
# ---------------------------------------------------------------------------

_PROGRAM_ALIASES: Dict[str, str] = {
    "BSCS": "BSCS", "CS": "BSCS", "COMPUTER SCIENCE": "BSCS", "COMSCI": "BSCS",
    "BSIT": "BSIT", "IT": "BSIT", "INFORMATION TECHNOLOGY": "BSIT",
    "BSIS": "BSIS", "IS": "BSIS", "INFORMATION SYSTEMS": "BSIS",
    "BTVTED": "BTVTED", "BTVTED-ICT": "BTVTED", "ICT": "BTVTED", "TVTED": "BTVTED",
}

PROGRAM_LABELS: Dict[str, str] = {
    "BSCS":   "BSCS (Computer Science)",
    "BSIT":   "BSIT (Information Technology)",
    "BSIS":   "BSIS (Information Systems)",
    "BTVTED": "BTVTED ICT",
}

#: Maps each program to its defining interests and skills keywords.
#: Used for profile-based alignment scoring.
PROGRAM_MAPPING: Dict[str, Dict[str, List[str]]] = {
    "BSCS": {
        "interests": [
            "Algorithms & Problem Solving", "Artificial Intelligence",
            "Software Engineering", "Data Structures", "Machine Learning",
        ],
        "skills": [
            "Programming", "Algorithm Design", "Logical thinking",
            "Debugging", "Mathematical analysis",
        ],
    },
    "BSIT": {
        "interests": [
            "Web Development", "Network Administration", "System Integration",
            "Cybersecurity", "Cloud Computing",
        ],
        "skills": [
            "Web development", "Network troubleshooting", "System administration",
            "Hardware setup", "Cybersecurity basics",
        ],
    },
    "BSIS": {
        "interests": [
            "Business Process Analysis", "Data Analytics", "Information Management",
            "Enterprise Systems", "Project Management",
        ],
        "skills": [
            "Data analysis", "Documentation", "Business communication",
            "System planning", "Critical thinking",
        ],
    },
    "BTVTED": {
        "interests": [
            "Technical Skills Development", "Teaching", "Industrial Tools",
            "Curriculum Design", "Applied Technologies",
        ],
        "skills": [
            "Technical teaching", "Hands-on skills", "Equipment handling",
            "Instructional planning", "Practical problem solving",
        ],
    },
}

# ---------------------------------------------------------------------------
# Synonyms (v5 Defense Ready)
# ---------------------------------------------------------------------------

SYNONYMS: Dict[str, str] = {
    "developer":  "development",
    "programmer": "programming",
    "coder":      "programming",
    "webdev":     "web",
    "frontend":   "web",
    "backend":    "web",
    "teacher":    "teaching",
    "instructor": "teaching",
    "analyst":    "analysis",
}

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def tokenize(text: str) -> List[str]:
    if not text:
        return []

    tokens = _TOKEN_RE.findall(text.lower())

    result: List[str] = []
    for t in tokens:
        value = SYNONYMS.get(t, t)
        if value is None:
            value = t
        result.append(str(value))

    return result


def normalize_text_list(items: List[str]) -> List[str]:
    """Returns a lowercase, stripped copy of *items*, excluding blank entries."""
    return [i.lower().strip() for i in items if i and i.strip()]


def normalize_program(raw: str) -> str:
    """Resolves a program string to its canonical four-letter code."""
    key = re.sub(r"\s+", " ", (raw or "").strip().upper())
    return _PROGRAM_ALIASES.get(key, key)


def program_label(program: str) -> str:
    """Returns a human-readable label for a canonical program code."""
    return PROGRAM_LABELS.get(
        normalize_program(program),
        normalize_program(program) or "Unknown Program",
    )


def cosine_sim_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Computes cosine similarity between two sparse TF-IDF vectors."""
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    na  = math.sqrt(sum(v * v for v in a.values()))
    nb  = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def l2_distance(a: List[float], b: List[float]) -> float:
    """Computes Euclidean (L2) distance between two equal-length vectors."""
    n = min(len(a), len(b))
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(n)))


# ---------------------------------------------------------------------------
# Fuzzy Match (v5 Defense Ready)
# ---------------------------------------------------------------------------


def fuzzy(a: str, b: str) -> float:
    """Returns a fuzzy string similarity ratio between two strings."""
    return SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------------------------
# BERT Similarity (v5 Defense Ready)
# ---------------------------------------------------------------------------


def bert_similarity(a: str, refs: List[str]) -> float:
    if not _BERT_AVAILABLE or BERT_MODEL is None or st_util is None:
        return 0.0

    if not a or not refs:
        return 0.0

    emb1 = BERT_MODEL.encode(a, convert_to_tensor=True)
    emb2 = BERT_MODEL.encode(refs, convert_to_tensor=True)

    scores = st_util.cos_sim(emb1, emb2)[0]
    return float(scores.max())


# ---------------------------------------------------------------------------
# Profile Scoring  (v5 Defense Ready — Token + Fuzzy + BERT)
# ---------------------------------------------------------------------------


def match_score(user_list: List[str], ref_list: List[str]) -> float:
    """
    Measures alignment between a user's profile items and a reference keyword
    list using three complementary signals:

        1. Token overlap  — exact token-level intersection
        2. Fuzzy match    — SequenceMatcher string similarity
        3. BERT similarity— semantic sentence-embedding cosine similarity

    The final per-item score is the *maximum* of all three signals,
    so even paraphrased or semantically related items are captured.

    Returns:
        float: Average match score in [0.0, 1.0], rounded to 4 decimal places.
    """
    if not user_list:
        return 0.0

    scores: List[float] = []
    for u in user_list:
        u_tokens   = set(tokenize(u))
        ref_tokens = [set(tokenize(r)) for r in ref_list]

        token_match  = any(u_tokens & r for r in ref_tokens)
        fuzzy_score  = max((fuzzy(u, r) for r in ref_list), default=0.0)
        bert_score   = bert_similarity(u, ref_list)

        final = max(1.0 if token_match else 0.0, fuzzy_score, bert_score)
        scores.append(final)

    return round(sum(scores) / len(scores), 4)


def score_profile_against_mapping(
    user_items: List[str],
    mapping_items: List[str],
) -> float:
    """
    Measures how many of the student's profile items overlap with a
    program's keyword list using token-level partial matching.

    Returns:
        float: Overlap ratio in [0.0, 1.0], normalised over mapping size.

    Note:
        For richer semantic matching (fuzzy + BERT), use ``match_score``
        instead — this function is retained for lightweight / legacy use.
    """
    if not user_items or not mapping_items:
        return 0.0

    user_token_sets    = [set(tokenize(item)) for item in user_items]
    mapping_token_sets = [set(tokenize(item)) for item in mapping_items]

    matched = sum(
        1 for u_tokens in user_token_sets
        if any(u_tokens & m_tokens for m_tokens in mapping_token_sets)
    )
    return matched / len(mapping_items)


def score_career_goals_against_mapping(
    career_goals: List[str],
    program: str,
) -> float:
    """
    Scores career goals against the combined interests and skills of a
    program using the enhanced Token + Fuzzy + BERT pipeline.

    Falls back to token-only matching when BERT is unavailable.

    Returns:
        float: Match ratio in [0.0, 1.0].
    """
    mapping  = PROGRAM_MAPPING.get(program, {})
    combined = mapping.get("interests", []) + mapping.get("skills", [])
    if not combined:
        return 0.0

    norm_goals = [g for g in career_goals if g.strip()]
    if not norm_goals:
        return 0.0

    # Use the rich match_score pipeline (Token + Fuzzy + BERT)
    return min(1.0, match_score(norm_goals, combined))


def compute_profile_scores(
    user_skills: List[str],
    user_interests: List[str],
    user_career_goals: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Computes per-program profile alignment scores across three dimensions:
    skills, interests, and career goals.

    Skills and Interests use the enhanced ``match_score`` pipeline
    (Token + Fuzzy + BERT) for maximum recall.

    Returns:
        Dict mapping each program code to a sub-score dictionary, e.g.::

            {
                "BSCS": {"skills": 0.80, "interests": 0.60, "career_goals": 0.40},
                ...
            }
    """
    norm_skills    = normalize_text_list(user_skills)
    norm_interests = normalize_text_list(user_interests)
    norm_goals     = normalize_text_list(user_career_goals)

    return {
        program: {
            "skills": round(
                match_score(norm_skills, mapping.get("skills", [])), 4
            ),
            "interests": round(
                match_score(norm_interests, mapping.get("interests", [])), 4
            ),
            "career_goals": round(
                score_career_goals_against_mapping(norm_goals, program), 4
            ),
        }
        for program, mapping in PROGRAM_MAPPING.items()
    }


# ---------------------------------------------------------------------------
# Weighted Scoring
# ---------------------------------------------------------------------------


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
    Computes the final weighted recommendation score per program.

    Quiz sub-score mapping (category → program):
        - Logic       → BSIS
        - Programming → BSCS
        - Networking  → BSIT
        - Design      → BTVTED

    The quiz component blends the overall percentage (50%) with the
    program-specific sub-score (50%) to reward both breadth and depth.

    Returns:
        Dict mapping each program code to a score in [0.0, 1.0].
    """
    quiz_total  = max(1, quiz_total)
    overall_pct = quiz_score / quiz_total

    program_quiz_map: Dict[str, float] = {
        "BSCS":   programming / quiz_total,
        "BSIT":   networking  / quiz_total,
        "BSIS":   logic       / quiz_total,
        "BTVTED": design      / quiz_total,
    }

    weighted: Dict[str, float] = {}
    for program in PROGRAM_MAPPING:
        quiz_component = (overall_pct * 0.5) + (program_quiz_map.get(program, 0.0) * 0.5)
        p = profile_scores.get(program, {})
        final = (
            quiz_component            * WEIGHT_QUIZ
            + p.get("skills", 0.0)    * WEIGHT_SKILLS
            + p.get("interests", 0.0) * WEIGHT_INTERESTS
            + p.get("career_goals", 0.0) * WEIGHT_CAREER_GOALS
        )
        weighted[program] = round(final, 6)

    return weighted


def pick_recommended_program(
    weighted_scores: Dict[str, float],
    preferred_program: str = "",
) -> str:
    """
    Selects the recommended program from weighted scores.

    Tie-breaking priority:
        1. Highest weighted score
        2. Student's preferred program (if tied)
        3. First alphabetical program (deterministic fallback)
    """
    if not weighted_scores:
        return "BSIT"

    max_score    = max(weighted_scores.values())
    top_programs = [p for p, s in weighted_scores.items() if s == max_score]

    if len(top_programs) == 1:
        return top_programs[0]

    preferred = normalize_program(preferred_program)
    if preferred and preferred in top_programs:
        return preferred

    return top_programs[0]


def compute_confidence(
    weighted_scores: Dict[str, float],
    recommended: str,
) -> int:
    """
    Estimates recommendation confidence as a percentage in [50, 97].

    Confidence is derived from the recommended program's weighted score,
    then adjusted based on its margin over the second-best program.
    A large margin boosts confidence; a small margin reduces it.

    Returns:
        int: Confidence percentage.
    """
    if not weighted_scores:
        return 50

    sorted_scores  = sorted(weighted_scores.values(), reverse=True)
    top, second    = sorted_scores[0], (sorted_scores[1] if len(sorted_scores) > 1 else 0.0)
    margin         = top - second
    raw_conf       = int(min(97, max(50, top * 100)))

    if margin >= 0.10:
        raw_conf = min(97, raw_conf + 5)
    elif margin <= 0.02:
        raw_conf = max(50, raw_conf - 5)

    return raw_conf


# ---------------------------------------------------------------------------
# GWA Computation
# ---------------------------------------------------------------------------


def compute_gwa_and_rating(score: int, total: int) -> Tuple[float, str, str, float]:
    """
    Converts a raw quiz score to an estimated GWA and descriptive rating.

    Returns:
        Tuple of (gwa, rating_label, remarks, percent_score).
    """
    total   = max(1, total)
    percent = (score / total) * 100.0

    gwa_table = [
        (96, 1.00), (94, 1.25), (92, 1.50), (89, 1.75), (87, 2.00),
        (84, 2.25), (82, 2.50), (79, 2.75), (75, 3.00),
    ]
    gwa = 5.00
    for threshold, value in gwa_table:
        if percent >= threshold:
            gwa = value
            break

    rating_map = [
        (1.50, "Excellent",
         "Your overall performance is outstanding, demonstrating a very strong academic foundation."),
        (2.25, "Very Good",
         "Your performance is commendable, showing a solid understanding of the subject matter."),
        (2.75, "Good",
         "Your performance is satisfactory, with evident strengths, though some areas require further improvement."),
        (3.00, "Satisfactory (Pass)",
         "You have met the minimum requirements. Focusing on weaker areas is recommended "
         "to improve your overall performance."),
        (float("inf"), "Needs Improvement",
         "Your performance indicates a need for improvement. Consistent practice and review "
         "are highly recommended to enhance your understanding."),
    ]

    rating, remarks = "Needs Improvement", ""
    for threshold, r, rem in rating_map:
        if gwa <= threshold:
            rating, remarks = r, rem
            break

    return round(gwa, 2), rating, remarks, round(percent, 1)


# ---------------------------------------------------------------------------
# Content-Based Filtering (CBF)
# ---------------------------------------------------------------------------


@dataclass
class CourseItem:
    """Represents a single academic course for CBF recommendation."""
    id:          int
    code:        str
    title:       str
    description: str
    program:     str
    level:       str
    tags:        str

    def as_text(self) -> str:
        """Concatenates all course fields into a single searchable string."""
        return f"{self.code} {self.title} {self.description} {self.program} {self.level} {self.tags}"


class CBFRecommender:
    """
    Content-Based Filtering recommender using TF-IDF vectorisation
    and cosine similarity for course-to-student matching.
    """

    def __init__(self) -> None:
        self._idf: Dict[str, float]              = {}
        self._course_vecs: Dict[int, Dict[str, float]] = {}
        self._fitted: bool                       = False

    def fit(self, courses: List[CourseItem]) -> None:
        """Builds the IDF index and TF-IDF vectors for all courses."""
        df: Dict[str, int]           = {}
        docs_tokens: Dict[int, List[str]] = {}

        for course in courses:
            tokens = tokenize(course.as_text())
            docs_tokens[course.id] = tokens
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1

        n_docs    = max(1, len(courses))
        self._idf = {
            t: math.log((n_docs + 1) / (cnt + 1)) + 1.0
            for t, cnt in df.items()
        }

        self._course_vecs = {}
        for course in courses:
            tf: Dict[str, int] = {}
            for token in docs_tokens[course.id]:
                tf[token] = tf.get(token, 0) + 1
            self._course_vecs[course.id] = {
                t: (1.0 + math.log(cnt)) * self._idf.get(t, 0.0)
                for t, cnt in tf.items()
            }

        self._fitted = True

    def _vectorize_query(self, text: str) -> Dict[str, float]:
        """Converts a free-text query to a TF-IDF vector using the fitted IDF."""
        tf: Dict[str, int] = {}
        for token in tokenize(text):
            tf[token] = tf.get(token, 0) + 1
        return {
            t: (1.0 + math.log(cnt)) * self._idf.get(t, 0.0)
            for t, cnt in tf.items()
        }

    def recommend(
        self,
        student_text: str,
        courses: List[CourseItem],
        top_n: int = 10,
        program_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Returns the top-N courses most similar to the student profile text,
        optionally filtered to a specific program.
        """
        if not courses:
            return []
        if not self._fitted:
            self.fit(courses)

        query_vec = self._vectorize_query(student_text)
        pf        = normalize_program(program_filter) if program_filter else None
        by_id     = {c.id: c for c in courses}

        scored: List[Tuple[int, float]] = [
            (c.id, cosine_sim_sparse(query_vec, self._course_vecs[c.id]))
            for c in courses
            if (not pf or normalize_program(c.program) == pf) and c.id in self._course_vecs
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            {
                "course_id": cid,
                "code":      by_id[cid].code,
                "title":     by_id[cid].title,
                "program":   normalize_program(by_id[cid].program),
                "score":     round(sim, 6),
            }
            for cid, sim in scored[:max(1, top_n)]
        ]


# ---------------------------------------------------------------------------
# K-Means Clustering
# ---------------------------------------------------------------------------


@dataclass
class StudentVector:
    """Stores a student's feature vector for K-Means clustering."""
    user_id:  int
    features: List[float]


class KMeansClusterer:
    """
    Lightweight K-Means clustering implementation for grouping students
    by academic performance and interest profile.
    """

    def __init__(self, k: int = 4, max_iter: int = 50, seed: int = 42) -> None:
        self.k          = k
        self.max_iter   = max_iter
        self.seed       = seed
        self.centroids: List[List[float]] = []
        self._fitted: bool = False
        self._dim: int     = 0

    def fit(self, data: List[StudentVector]) -> None:
        """Fits the clusterer to a list of student feature vectors."""
        if not data:
            return self._reset()

        random.seed(self.seed)
        points = [sv.features for sv in data if sv.features]
        if not points:
            return self._reset()

        dim    = len(points[0])
        points = [p for p in points if len(p) == dim]
        if not points:
            return self._reset()

        self._dim  = dim
        init_k     = min(self.k, len(points))
        self.centroids = [p[:] for p in random.sample(points, k=init_k)]
        while len(self.centroids) < self.k:
            self.centroids.append(points[0][:])

        for _ in range(self.max_iter):
            clusters: List[List[List[float]]] = [[] for _ in range(self.k)]
            for p in points:
                clusters[self._nearest_centroid_index(p)].append(p)

            new_centroids = [
                self._mean_vector(cluster) if cluster
                else points[random.randint(0, len(points) - 1)][:]
                for cluster in clusters
            ]

            shift = sum(l2_distance(a, b) for a, b in zip(self.centroids, new_centroids))
            self.centroids = new_centroids
            if shift < 1e-6:
                break

        self._fitted = True

    def predict(self, features: List[float]) -> int:
        """Returns the cluster index for a given feature vector."""
        if not self._fitted or not self.centroids or not features:
            return 0
        if self._dim and len(features) != self._dim:
            return 0
        return self._nearest_centroid_index(features)

    def _reset(self) -> None:
        self.centroids, self._fitted, self._dim = [], False, 0

    def _nearest_centroid_index(self, point: List[float]) -> int:
        return min(
            range(len(self.centroids)),
            key=lambda i: l2_distance(point, self.centroids[i]),
        )

    @staticmethod
    def _mean_vector(points: List[List[float]]) -> List[float]:
        dim = len(points[0])
        return [sum(p[j] for p in points) / len(points) for j in range(dim)]


# ---------------------------------------------------------------------------
# Feature Vector Builder
# ---------------------------------------------------------------------------


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
    Constructs a 7-dimensional numeric feature vector representing a student's
    academic profile for use in K-Means clustering.

    Dimensions:
        [overall%, logic%, programming%, networking%, design%,
         interest_token_count, behavior_score]
    """
    total = max(1, total)
    return [
        (score       / total) * 100.0,
        (logic       / total) * 100.0,
        (programming / total) * 100.0,
        (networking  / total) * 100.0,
        (design      / total) * 100.0,
        float(len(tokenize(interests_text))),
        float(behavior_score),
    ]


# ---------------------------------------------------------------------------
# Student Query Text (for CBF)
# ---------------------------------------------------------------------------


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
    """
    Constructs a rich free-text query string for CBF matching by combining
    quiz strengths, profile data, and preferred program keywords.
    """
    total     = max(1, total)
    threshold = max(1, int(round(total * 0.05)))

    strength_terms: List[str] = []
    if strengths.get("programming", 0) >= threshold:
        strength_terms += ["programming", "software", "coding", "algorithms"]
    if strengths.get("networking", 0) >= threshold:
        strength_terms += ["networking", "systems", "infrastructure", "security"]
    if strengths.get("logic", 0) >= threshold:
        strength_terms += ["analysis", "systems analysis", "requirements", "database"]
    if strengths.get("design", 0) >= threshold:
        strength_terms += ["design", "multimedia", "instructional", "teaching"]

    preferred_keyword_map: Dict[str, str] = {
        "BSCS":   "computer science programming software development algorithms",
        "BSIT":   "information technology networking systems infrastructure support",
        "BSIS":   "information systems analysis database business process",
        "BTVTED": "btvted ict multimedia design educational technology teaching",
    }
    preferred_tokens = preferred_keyword_map.get(normalize_program(preferred_program), "")

    extra_tokens = " ".join(
        normalize_text_list(
            (user_skills or []) + (user_interests_list or []) + (user_career_goals_list or [])
        )
    )

    return (
        f"{interests} {career_goals} {strand} "
        f"{preferred_tokens} {' '.join(strength_terms)} {extra_tokens}"
    ).strip()


# ---------------------------------------------------------------------------
# Explainable AI (XAI) — Report Builder
# ---------------------------------------------------------------------------


def build_program_scores_summary(
    weighted_scores: Dict[str, float],
    recommended: str,
) -> str:
    """
    Returns a clean list of program scores with no sub-score breakdown.
    Format: BSIT (Information Technology) 32.5% ✅ Recommended
    """
    lines = []
    for prog, ws in sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True):
        marker = " ✅ Recommended" if normalize_program(prog) == normalize_program(recommended) else ""
        lines.append(f"   {program_label(prog)} {ws * 100:.1f}%{marker}")
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
    weighted_scores: Optional[Dict[str, float]] = None,
    profile_scores: Optional[Dict[str, Dict[str, float]]] = None,
    ai_explanation: str = "",
    course_recommendations: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Assembles the complete, human-readable recommendation report.

    Sections (in order):
        1. Assessment Summary   — GWA, score, rating, remarks
        2. Recommendation       — Preferred & recommended program
        3. Explainable AI       — LLM-generated single-paragraph reason
        4. Program Scores       — Clean percentage list, no formula breakdown
        5. Suggested Courses    — CBF course recommendations
    """
    pct = (score / max(1, total)) * 100.0

    preferred_text = (
        program_label(preferred_program)
        if preferred_program else "Not specified"
    )

    # ── Program Scores Section ────────────────────────────────────────────────
    scores_section = ""
    if weighted_scores:
        scores_section = (
            "\n\n Recommendation Scores\n"
            + build_program_scores_summary(weighted_scores, recommended_program)
        )

    # ── XAI Section ───────────────────────────────────────────────────────────
    ai_section = (
        f"\n Explainable AI\n"
        f"   {ai_explanation}\n"
        if ai_explanation else ""
    )

    # ── Suggested Courses Section ─────────────────────────────────────────────
    courses_section = ""
    if course_recommendations:
        course_lines = [
            f"   {i + 1}. [{c['code']}] {c['title']}  (match: {c['score'] * 100:.1f}%)"
            for i, c in enumerate(course_recommendations)
        ]
        courses_section = (
            "\n\n Suggested Courses\n"
            + "\n".join(course_lines)
        )

    return (
        f"{'=' * 60}\n"
        f"       ACADEMIC PROGRAM RECOMMENDATION REPORT\n"
        f"{'=' * 60}\n\n"
        f" Assessment Summary\n"
        f"   Rating             : {rating} (Est. GWA: {gwa})\n"
        f"   Score              : {score}/{total} ({pct:.1f}%)\n"
        f"   Remarks            : {gwa_remarks}\n\n"
        f" Recommendation\n"
        f"   Preferred Program  : {preferred_text}\n"
        f"   Recommended Program: {program_label(recommended_program)}\n"
        f"   Confidence         : {confidence}%\n"
        f"{scores_section}"
        f"{ai_section}"
        f"{courses_section}\n"
        f"{'=' * 60}"
    )


# ---------------------------------------------------------------------------
# Explainable AI (XAI) — LLM Integration via OpenRouter
# ---------------------------------------------------------------------------


def build_ai_explanation_prompt(
    recommended_program: str,
    preferred_program: str,
    weighted_scores: Dict[str, float],
    profile_scores: Dict[str, Dict[str, float]],
    user_skills: List[str],
    user_interests: List[str],
    user_career_goals: List[str],
    percent: float,
) -> str:
    """
    Constructs a structured prompt for the LLM to generate a single-paragraph,
    professional explanation of the recommendation result.

    The paragraph must include:
      - Student's skills, interests, and career goals
      - Preferred program vs recommended program
      - Professional reason why the recommended program suits the student
    """
    rec_label  = program_label(recommended_program)
    pref_label = program_label(preferred_program) if preferred_program else "Not specified"

    scores_table = "\n".join(
        f"  - {program_label(prog)}: {score * 100:.1f}%"
        for prog, score in sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
    )

    rec_profile = profile_scores.get(normalize_program(recommended_program), {})

    skills_str    = ", ".join(user_skills)        if user_skills        else "not specified"
    interests_str = ", ".join(user_interests)     if user_interests     else "not specified"
    goals_str     = ", ".join(user_career_goals)  if user_career_goals  else "not specified"

    return f"""You are a professional academic program advisor for an IT college.

A student has completed an academic readiness assessment and the system has produced the following results:

Preferred Program  : {pref_label}
Recommended Program: {rec_label}
Overall Quiz Score : {percent:.1f}%

Student Profile:
  - Skills        : {skills_str}
  - Interests     : {interests_str}
  - Career Goals  : {goals_str}

Program Scores:
{scores_table}

Profile Alignment for {rec_label}:
  - Skills Match      : {int(rec_profile.get('skills', 0) * 100)}%
  - Interests Match   : {int(rec_profile.get('interests', 0) * 100)}%
  - Career Goals Match: {int(rec_profile.get('career_goals', 0) * 100)}%

TASK:
Write exactly ONE professional paragraph (4–5 sentences) that:
  1. Mentions the student's skills, interests, and career goals naturally.
  2. States the preferred program and recommended program clearly.
  3. Provides a professional reason why {rec_label} is the most suitable program for this student based on the data.
  4. Does NOT give advice or commands — only provide reasoned justification.
  5. Sounds like a formal academic program evaluator, not a life coach.

Do not use bullet points, headers, or multiple paragraphs. Output only the paragraph text.
"""


def generate_ai_explanation(
    prompt: str,
    db: Any = None,
    user_id: Optional[int] = None,
    conversation_id: Optional[str] = None,
) -> str:
    """
    Calls the OpenRouter LLM API to generate a human-readable,
    professional single-paragraph explanation for the recommendation.
    """
    try:
        import openai

        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            logger.warning("OPENROUTER_API_KEY is not set. Skipping AI explanation.")
            return ""

        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        model = os.getenv("OPENROUTER_MODEL", "qwen/qwen3-235b-a22b:free")

        logger.info(
            "Generating AI explanation via OpenRouter | user_id=%s | model=%s",
            user_id, model,
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a formal academic program evaluator.\n"
                        "Write only one paragraph (4–5 sentences).\n"
                        "Only use the provided data. Do not invent scores, skills, or results.\n"
                        "Do not give advice or commands. Do not use bullet points or headers.\n"
                        "Do not contradict the recommendation."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.5,
        )

        explanation = response.choices[0].message.content or ""
        logger.info("AI explanation generated successfully for user_id=%s", user_id)
        return explanation.strip()

    except ImportError:
        logger.error("openai package is not installed. Run: pip install openai")
        return ""
    except Exception as exc:  # noqa: BLE001
        logger.error("AI explanation generation failed for user_id=%s: %s", user_id, exc)
        return ""


# ---------------------------------------------------------------------------
# Main Recommendation Entry Point
# ---------------------------------------------------------------------------


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
    user_skills: Optional[List[str]] = None,
    user_interests: Optional[List[str]] = None,
    user_career_goals: Optional[List[str]] = None,
    historical_students: Optional[List[StudentVector]] = None,
    courses: Optional[List[CourseItem]] = None,
    top_n_courses: int = 10,
    enable_ai_explanation: bool = True,
    db: Any = None,
    conversation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main entry point for the hybrid academic program recommendation system.

    Pipeline:
        1. K-Means Clustering
        2. Profile Scoring  (Token + Fuzzy + BERT)
        3. Weighted Recommendation Formula
        4. GWA & Rating
        5. Explainable AI (XAI) via LLM
        6. CBF Course Recommendations
        7. Final Report Message
    """
    _skills    = user_skills        or []
    _interests = user_interests     or []
    _goals     = user_career_goals  or []

    # ── Step 1: K-Means Clustering ──────────────────────────────────────────
    feature_vec = build_student_feature_vector(
        score=score, total=total,
        logic=logic, programming=programming,
        networking=networking, design=design,
        interests_text=interests,
        behavior_score=behavior_score,
    )

    cluster_id = 0
    if historical_students:
        km = KMeansClusterer(k=4)
        km.fit(historical_students)
        cluster_id = km.predict(feature_vec)

    logger.info("user_id=%s assigned to cluster_id=%s", user_id, cluster_id)

    # ── Step 2: Profile Scoring (Token + Fuzzy + BERT) ─────────────────────
    profile_scores = compute_profile_scores(
        user_skills=_skills,
        user_interests=_interests,
        user_career_goals=_goals,
    )

    # ── Step 3: Weighted Scoring & Program Selection ─────────────────────────
    weighted_scores = compute_weighted_scores(
        quiz_score=score, quiz_total=total,
        logic=logic, programming=programming,
        networking=networking, design=design,
        profile_scores=profile_scores,
    )
    recommended_program = pick_recommended_program(weighted_scores, preferred_program)
    confidence          = compute_confidence(weighted_scores, recommended_program)

    logger.info(
        "user_id=%s → recommended=%s | confidence=%d%%",
        user_id, recommended_program, confidence,
    )

    # ── Step 4: GWA & Rating ────────────────────────────────────────────────
    gwa, rating_label, gwa_remarks, pct = compute_gwa_and_rating(score=score, total=total)

    # ── Step 5: Explainable AI (XAI) ────────────────────────────────────────
    ai_explanation = ""
    if enable_ai_explanation:
        ai_prompt = build_ai_explanation_prompt(
            recommended_program=recommended_program,
            preferred_program=preferred_program,
            weighted_scores=weighted_scores,
            profile_scores=profile_scores,
            user_skills=_skills,
            user_interests=_interests,
            user_career_goals=_goals,
            percent=pct,
        )
        ai_explanation = generate_ai_explanation(
            prompt=ai_prompt,
            db=db,
            user_id=user_id,
            conversation_id=conversation_id,
        )

    # ── Step 6: CBF Course Recommendations ───────────────────────────────────
    cbf_results: List[Dict[str, Any]] = []
    if courses:
        student_text = build_student_query_text(
            interests=interests,
            career_goals=career_goals,
            strand=strand,
            strengths={
                "logic": logic, "programming": programming,
                "networking": networking, "design": design,
            },
            total=total,
            preferred_program=preferred_program,
            user_skills=_skills,
            user_interests_list=_interests,
            user_career_goals_list=_goals,
        )
        normalised_courses = [
            CourseItem(
                id=c.id, code=c.code, title=c.title,
                description=c.description,
                program=normalize_program(c.program),
                level=c.level, tags=c.tags,
            )
            for c in courses
        ]
        cbf = CBFRecommender()
        cbf.fit(normalised_courses)
        cbf_results = cbf.recommend(
            student_text=student_text,
            courses=normalised_courses,
            top_n=top_n_courses,
            program_filter=recommended_program,
        )

    # ── Step 7: Final Report Message ─────────────────────────────────────────
    final_message = build_explainable_message(
        gwa=gwa,
        rating=rating_label,
        gwa_remarks=gwa_remarks,
        preferred_program=preferred_program,
        recommended_program=recommended_program,
        confidence=confidence,
        score=score,
        total=total,
        weighted_scores=weighted_scores,
        profile_scores=profile_scores,
        ai_explanation=ai_explanation,
        course_recommendations=cbf_results,
    )

    return {
        "user_id":               user_id,
        "cluster_id":            cluster_id,
        "percent_score":         pct,
        "gwa":                   gwa,
        "rating":                rating_label,
        "gwa_remarks":           gwa_remarks,
        "preferred_program":     normalize_program(preferred_program) if preferred_program else "",
        "recommended_program":   recommended_program,
        "confidence":            confidence,
        "weighted_scores":       weighted_scores,
        "profile_scores":        profile_scores,
        "message":               final_message,
        "ai_explanation":        ai_explanation,
        "course_recommendations": cbf_results,
    }


# ---------------------------------------------------------------------------
# Backward-Compatible Legacy Entry Point
# ---------------------------------------------------------------------------


def recommend_program(
    score: int,
    total: int,
    logic: int = 0,
    programming: int = 0,
    networking: int = 0,
    design: int = 0,
) -> Tuple[str, int, str]:
    """
    Legacy entry point for quiz-only recommendations (no profile inputs).

    Returns:
        Tuple of (recommended_program_code, confidence_percent, dummy_rationale).
    """
    dummy_profile = compute_profile_scores([], [], [])
    weighted      = compute_weighted_scores(
        quiz_score=score, quiz_total=total,
        logic=logic, programming=programming,
        networking=networking, design=design,
        profile_scores=dummy_profile,
    )
    program    = pick_recommended_program(weighted)
    confidence = compute_confidence(weighted, program)
    return program, confidence, ""