"""
Tests for the full recommendation pipeline.

Tests:
  - Output format (exact required fields)
  - Pipeline integration
  - Edge cases
"""

import json
import pytest
import numpy as np
from pathlib import Path

from engine.embeddings import create_embeddings


@pytest.fixture(scope="module")
def sample_catalogue():
    return [
        {
            "name": "Python (New)",
            "url": "https://www.shl.com/products/product-catalog/view/python-new/",
            "remote_support": "Yes",
            "adaptive_support": "No",
            "test_type": ["Knowledge & Skills"],
            "description": "Tests Python programming: OOP, data.",
            "duration": 30,
        },
        {
            "name": "Java 8",
            "url": "https://www.shl.com/products/product-catalog/view/java-8/",
            "remote_support": "Yes",
            "adaptive_support": "No",
            "test_type": ["Knowledge & Skills"],
            "description": "Tests Java 8 features.",
            "duration": 35,
        },
        {
            "name": "OPQ32r Personality",
            "url": "https://www.shl.com/products/product-catalog/view/opq32r/",
            "remote_support": "Yes",
            "adaptive_support": "No",
            "test_type": ["Personality & Behavior"],
            "description": "Personality characteristics for workplace.",
            "duration": 25,
        },
        {
            "name": "Verify G+",
            "url": "https://www.shl.com/products/product-catalog/view/verify-g/",
            "remote_support": "Yes",
            "adaptive_support": "Yes",
            "test_type": ["Ability & Aptitude"],
            "description": "Cognitive ability assessment.",
            "duration": 36,
        },
        {
            "name": "Customer Sim",
            "url": "https://www.shl.com/products/product-catalog/view/cust-sim/",
            "remote_support": "Yes",
            "adaptive_support": "No",
            "test_type": ["Simulations"],
            "description": "Customer service simulation.",
            "duration": 45,
        },
    ]


@pytest.fixture(scope="module")
def setup_pipeline(sample_catalogue, tmp_path_factory):
    """Set up the search module with test data."""
    import engine.search as search_mod

    embeddings = create_embeddings(sample_catalogue, show_progress=False)
    search_mod._catalogue = sample_catalogue
    search_mod._embeddings = embeddings

    return search_mod


# Required output fields per the spec
REQUIRED_FIELDS = {"url", "name", "adaptive_support", "description",
                   "duration", "remote_support", "test_type"}


class TestRecommendOutput:
    def test_has_all_required_fields(self, setup_pipeline):
        from engine.recommender import recommend
        results = recommend("Python programming test", top_k=3)
        assert len(results) > 0
        for r in results:
            assert set(r.keys()) == REQUIRED_FIELDS, (
                f"Missing fields: {REQUIRED_FIELDS - set(r.keys())}"
            )

    def test_test_type_is_array(self, setup_pipeline):
        from engine.recommender import recommend
        results = recommend("any test", top_k=5)
        for r in results:
            assert isinstance(r["test_type"], list), (
                f"test_type should be list, got {type(r['test_type'])}"
            )

    def test_duration_is_int_or_null(self, setup_pipeline):
        from engine.recommender import recommend
        results = recommend("test", top_k=5)
        for r in results:
            assert r["duration"] is None or isinstance(r["duration"], int)

    def test_url_is_shl_link(self, setup_pipeline):
        from engine.recommender import recommend
        results = recommend("test", top_k=5)
        for r in results:
            assert r["url"].startswith("https://www.shl.com/")

    def test_returns_max_10(self, setup_pipeline):
        from engine.recommender import recommend
        results = recommend("test", top_k=10)
        assert len(results) <= 10

    def test_returns_at_least_1(self, setup_pipeline):
        from engine.recommender import recommend
        results = recommend("Python test", top_k=10)
        assert len(results) >= 1

    def test_python_query_relevance(self, setup_pipeline):
        from engine.recommender import recommend
        results = recommend("Python programming test", top_k=3)
        assert results[0]["name"] == "Python (New)"

    def test_empty_query(self, setup_pipeline):
        from engine.recommender import recommend
        results = recommend("", top_k=5)
        assert isinstance(results, list)

    def test_remote_support_values(self, setup_pipeline):
        from engine.recommender import recommend
        results = recommend("test", top_k=5)
        for r in results:
            assert r["remote_support"] in ("Yes", "No")

    def test_adaptive_support_values(self, setup_pipeline):
        from engine.recommender import recommend
        results = recommend("test", top_k=5)
        for r in results:
            assert r["adaptive_support"] in ("Yes", "No")
