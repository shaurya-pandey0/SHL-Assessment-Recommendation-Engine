"""
Vector similarity search for SHL assessments.

Uses cosine similarity (dot product on normalized vectors) to find
the most semantically similar assessments to a given query.
"""

import logging
from pathlib import Path

import numpy as np

from engine.embeddings import embed_query, load_embeddings

logger = logging.getLogger(__name__)

# Module-level state — loaded once
_catalogue = None
_embeddings = None


def _ensure_loaded():
    """Load catalogue and embeddings if not already loaded."""
    global _catalogue, _embeddings
    if _catalogue is None or _embeddings is None:
        _catalogue, _embeddings = load_embeddings()
        logger.info(f"Search engine loaded: {len(_catalogue)} assessments")


def vector_search(query: str, top_k: int = 20) -> list[dict]:
    """
    Search for assessments semantically similar to the query.

    Args:
        query: Natural language search query.
        top_k: Number of results to return.

    Returns:
        List of assessment dicts with added 'score' field,
        sorted by descending similarity.
    """
    _ensure_loaded()

    # Embed the query
    query_vec = embed_query(query)

    # Cosine similarity = dot product (both vectors are L2-normalized)
    similarities = np.dot(_embeddings, query_vec)

    # Get top-K indices
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # Build results
    results = []
    for idx in top_indices:
        assessment = _catalogue[idx].copy()
        assessment["score"] = float(similarities[idx])
        results.append(assessment)

    return results


def reload():
    """Force reload of catalogue and embeddings."""
    global _catalogue, _embeddings
    _catalogue = None
    _embeddings = None
    _ensure_loaded()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_queries = [
        "Python programming assessment under 30 minutes",
        "Leadership personality assessment",
        "cognitive ability test for analysts",
        "Java developer test",
        "customer service simulation",
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 60)
        results = vector_search(query, top_k=5)
        for i, r in enumerate(results, 1):
            duration = r.get("duration", "N/A")
            test_type = ", ".join(r.get("test_type", []))
            print(f"  {i}. {r['name']:<40} | {duration:>4} min | {test_type:<25} | {r['score']:.3f}")
