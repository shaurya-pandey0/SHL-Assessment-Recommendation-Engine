"""
Vector similarity search engine for SHL assessments.

Uses cosine similarity (via dot product on normalized vectors) to find
the most semantically similar assessments to a given query.

At ~400 assessments and 384 dimensions, this is a tiny dataset —
a simple numpy dot product is faster than FAISS setup overhead.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from engine.embeddings import embed_query, load_embeddings

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


class VectorSearchEngine:
    """
    Vector search engine over SHL assessment embeddings.

    Performs cosine similarity search using pre-normalized embeddings.
    Since embeddings are L2-normalized at creation time, cosine similarity
    reduces to a simple dot product.
    """

    def __init__(
        self,
        catalogue_path: Path = None,
        embeddings_path: Path = None,
    ):
        if catalogue_path is None:
            catalogue_path = DATA_DIR / "catalogue.json"
        if embeddings_path is None:
            embeddings_path = DATA_DIR / "embeddings.npy"

        # Load catalogue
        with open(catalogue_path, "r", encoding="utf-8") as f:
            self.catalogue = json.load(f)

        # Load embeddings
        self.embeddings = load_embeddings(embeddings_path)

        if len(self.catalogue) != self.embeddings.shape[0]:
            raise ValueError(
                f"Catalogue ({len(self.catalogue)}) and embeddings "
                f"({self.embeddings.shape[0]}) size mismatch!"
            )

        logger.info(
            f"VectorSearchEngine initialized with {len(self.catalogue)} assessments, "
            f"embedding dim={self.embeddings.shape[1]}"
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Search for assessments semantically similar to the query.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.

        Returns:
            List of assessment dicts with added 'score' field,
            sorted by descending similarity score.
        """
        # Embed the query
        query_embedding = embed_query(query)

        # Compute cosine similarity via dot product (embeddings are pre-normalized)
        similarities = np.dot(self.embeddings, query_embedding)

        # Get top-K indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Build results
        results = []
        for idx in top_indices:
            assessment = self.catalogue[idx].copy()
            assessment["score"] = float(similarities[idx])
            results.append(assessment)

        return results

    def search_with_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Search using a pre-computed query embedding.

        Useful when you want to embed once and search multiple times,
        or when testing with synthetic embeddings.
        """
        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            assessment = self.catalogue[idx].copy()
            assessment["score"] = float(similarities[idx])
            results.append(assessment)

        return results

    def get_assessment_by_name(self, name: str) -> Optional[dict]:
        """Find an assessment by exact name match."""
        for assessment in self.catalogue:
            if assessment["name"] == name:
                return assessment
        return None

    def get_all_test_types(self) -> list[str]:
        """Get all unique test type values in the catalogue."""
        types = set()
        for a in self.catalogue:
            for t in a.get("test_types", []):
                types.add(t)
        return sorted(types)

    @property
    def size(self) -> int:
        """Number of assessments in the catalogue."""
        return len(self.catalogue)


# Module-level singleton for convenience
_engine: Optional[VectorSearchEngine] = None


def get_search_engine() -> VectorSearchEngine:
    """Get or create the global search engine singleton."""
    global _engine
    if _engine is None:
        _engine = VectorSearchEngine()
    return _engine


def search(query: str, top_k: int = 10) -> list[dict]:
    """Convenience function for quick searches."""
    engine = get_search_engine()
    return engine.search(query, top_k=top_k)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Quick test
    print("\n" + "=" * 80)
    print("VECTOR SEARCH TEST")
    print("=" * 80)

    test_queries = [
        "Python programming assessment under 30 minutes",
        "Leadership personality assessment",
        "cognitive ability test for analysts",
        "Java developer test",
        "customer service skills assessment",
    ]

    engine = VectorSearchEngine()

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 60)
        results = engine.search(query, top_k=5)
        for i, r in enumerate(results, 1):
            duration = r.get("duration", "N/A")
            print(
                f"  {i}. {r['name']:<45} "
                f"| {duration:>4}min "
                f"| {r['test_type']:<25} "
                f"| score={r['score']:.3f}"
            )
