"""
Full recommendation pipeline.

Chains: query parsing → vector search → re-rank/balance → format output.

Output format (exact fields required by SHL):
{
    "url": str,
    "name": str,
    "adaptive_support": "Yes"/"No",
    "description": str,
    "duration": int or null,
    "remote_support": "Yes"/"No",
    "test_type": ["Knowledge & Skills", ...]  # array
}
"""

import logging
from typing import Optional

from engine.search import vector_search
from engine.query_parser import parse_query
from engine.reranker import rerank

logger = logging.getLogger(__name__)


def recommend(query: str, top_k: int = 10) -> list[dict]:
    """
    Full recommendation pipeline.

    Args:
        query: Natural language query or job description text.
        top_k: Number of results to return (max 10).

    Returns:
        List of formatted recommendation dicts.
    """
    # 1. Parse query to extract structured intent
    parsed = parse_query(query)
    logger.info(f"Parsed query: {parsed}")

    # 2. Vector search — get 20 candidates
    candidates = vector_search(query, top_k=20)

    # 3. Re-rank and balance
    results = rerank(candidates, parsed, top_k=top_k)

    # 4. Format output with exact required fields
    return [
        {
            "url": r.get("url", ""),
            "name": r.get("name", ""),
            "adaptive_support": r.get("adaptive_support", "No"),
            "description": r.get("description", ""),
            "duration": r.get("duration"),
            "remote_support": r.get("remote_support", "No"),
            "test_type": r.get("test_type", []),  # already an array
        }
        for r in results
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_queries = [
        "Python programming assessment under 30 minutes",
        "Java developer who collaborates with external teams",
        "Leadership personality assessment for managers",
        "Cognitive ability test for analysts, max 40 minutes",
        "Customer service simulation under 60 minutes",
    ]

    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: '{query}'")
        print(f"{'='*80}")
        results = recommend(query)
        for i, r in enumerate(results, 1):
            test_type = ", ".join(r["test_type"])
            print(
                f"  {i}. {r['name']:<40} "
                f"| {r['duration'] or 'N/A':>4} min "
                f"| {test_type:<25} "
                f"| remote={r['remote_support']}"
            )
