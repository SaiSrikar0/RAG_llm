"""Retrieval logic, scoring, and threshold category handling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalResult:
    context_chunks: list[str]
    score: float
    category: str


def similarity_from_distance(distance: float) -> float:
    """Convert cosine distance (0 is best) to similarity score (1 is best)."""
    score = 1.0 - float(distance)
    return max(0.0, min(1.0, score))


def categorize_similarity(score: float) -> str:
    if score >= 0.9:
        return "High Match"
    if 0.5 <= score < 0.9:
        return "Similar"
    return "Not Found"


def parse_results(raw: dict) -> RetrievalResult:
    docs = (raw.get("documents") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]

    if not docs:
        return RetrievalResult(context_chunks=[], score=0.0, category="Not Found")

    best_score = similarity_from_distance(distances[0]) if distances else 0.0
    category = categorize_similarity(best_score)
    return RetrievalResult(context_chunks=docs, score=best_score, category=category)
