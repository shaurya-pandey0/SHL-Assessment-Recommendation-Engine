"""
Full recommendation pipeline.

Chains together: query parsing → vector search → filtering → re-ranking.

The pipeline:
1. Parse constraints from query (duration, test type, remote)
2. Vector search to find top-20 semantically similar assessments
3. Apply hard filters (duration, remote)
4. Boost matching test types (+0.3 to similarity score)
5. Re-rank and return top-K
"""

import logging
from typing import Optional

from engine.query_parser import parse_query
from engine.search import VectorSearchEngine, get_search_engine

logger = logging.getLogger(__name__)

# Boost applied to similarity score for test type matches
TEST_TYPE_BOOST = 0.3

# Number of candidates to fetch before filtering
CANDIDATE_POOL_SIZE = 20


def recommend(
    query: str,
    top_k: int = 10,
    engine: VectorSearchEngine = None,
) -> list[dict]:
    """
    Full recommendation pipeline.

    Args:
        query: Natural language query or job description.
        top_k: Number of results to return.
        engine: Optional search engine instance (uses global singleton if None).

    Returns:
        List of recommendation dicts sorted by relevance.
    """
    if engine is None:
        engine = get_search_engine()

    # 1. Parse constraints from query
    constraints = parse_query(query)
    logger.info(f"Parsed constraints: {constraints}")

    # 2. Vector search — get a larger candidate pool
    pool_size = max(CANDIDATE_POOL_SIZE, top_k * 2)
    candidates = engine.search(query, top_k=pool_size)

    # 3. Apply filters and re-rank
    results = apply_filters(candidates, constraints)

    # 4. Format and return top-K
    return format_results(results[:top_k])


def apply_filters(
    candidates: list[dict],
    constraints: dict,
) -> list[dict]:
    """
    Apply hard filters and soft boosts based on parsed constraints.

    Hard filters (remove non-matching):
        - max_duration: filter out assessments exceeding the limit
        - remote_required: filter out non-remote assessments

    Soft boosts (re-rank):
        - test_types: add +0.3 to score for matching types
    """
    filtered = []

    max_duration = constraints.get("max_duration")
    desired_types = constraints.get("test_types", [])
    remote_required = constraints.get("remote_required")

    for candidate in candidates:
        # Hard filter: duration
        if max_duration is not None:
            duration = candidate.get("duration")
            if duration is not None and duration > max_duration:
                continue

        # Hard filter: remote testing
        if remote_required and candidate.get("remote_support") != "Yes":
            continue

        # Soft boost: test type matching
        boosted_score = candidate.get("score", 0.0)
        if desired_types:
            candidate_types = candidate.get("test_types", [])
            if any(dt in candidate_types for dt in desired_types):
                boosted_score += TEST_TYPE_BOOST

        candidate = candidate.copy()
        candidate["boosted_score"] = boosted_score
        filtered.append(candidate)

    # Sort by boosted score (descending)
    filtered.sort(key=lambda x: x.get("boosted_score", 0), reverse=True)

    return filtered


def format_results(results: list[dict]) -> list[dict]:
    """
    Format results into the required output schema.

    Each result contains:
        - assessment_name: str
        - url: str
        - remote_support: "Yes" / "No"
        - adaptive_irt: "Yes" / "No"
        - duration: int or null
        - test_type: str
    """
    formatted = []
    for r in results:
        formatted.append({
            "assessment_name": r.get("name", ""),
            "url": r.get("url", ""),
            "remote_support": r.get("remote_support", "No"),
            "adaptive_irt": r.get("adaptive_irt", "No"),
            "duration": r.get("duration"),
            "test_type": r.get("test_type", "Unknown"),
        })
    return formatted


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_queries = [
        "Python programming assessment under 30 minutes",
        "Leadership personality assessment for managers",
        "Cognitive ability test for analysts, max 40 minutes",
        "Java developer test, online",
        "Customer service simulation under 60 minutes",
    ]

    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: '{query}'")
        print(f"{'='*80}")
        results = recommend(query)
        for i, r in enumerate(results, 1):
            print(
                f"  {i}. {r['assessment_name']:<45} "
                f"| {r['duration'] or 'N/A':>4} min "
                f"| {r['test_type']:<25} "
                f"| remote={r['remote_support']}"
            )
