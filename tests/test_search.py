"""
Tests for vector search.

Tests:
  - Search returns correct number of results
  - Results sorted by score
  - Semantic relevance (Python query → Python result)
  - Search result fields
"""

import json
import pytest
import numpy as np
from pathlib import Path

from engine.embeddings import create_embeddings


# Use the new simple module-level search API
# We need to monkeypatch the module globals

@pytest.fixture(scope="module")
def sample_catalogue():
    return [
        {
            "name": "Python (New)",
            "url": "https://www.shl.com/products/product-catalog/view/python-new/",
            "remote_support": "Yes",
            "adaptive_support": "No",
            "test_type": ["Knowledge & Skills"],
            "description": "Tests Python programming: OOP, data structures, functions.",
            "duration": 30,
        },
        {
            "name": "Java 8",
            "url": "https://www.shl.com/products/product-catalog/view/java-8/",
            "remote_support": "Yes",
            "adaptive_support": "No",
            "test_type": ["Knowledge & Skills"],
            "description": "Tests Java 8: streams, lambda, concurrency.",
            "duration": 35,
        },
        {
            "name": "Verify G+ Cognitive",
            "url": "https://www.shl.com/products/product-catalog/view/verify-g-plus/",
            "remote_support": "Yes",
            "adaptive_support": "Yes",
            "test_type": ["Ability & Aptitude"],
            "description": "Cognitive ability: numerical, verbal, inductive reasoning.",
            "duration": 36,
        },
        {
            "name": "OPQ32r Personality",
            "url": "https://www.shl.com/products/product-catalog/view/opq32r/",
            "remote_support": "Yes",
            "adaptive_support": "No",
            "test_type": ["Personality & Behavior"],
            "description": "32 personality characteristics for workplace assessment.",
            "duration": 25,
        },
        {
            "name": "Customer Service Sim",
            "url": "https://www.shl.com/products/product-catalog/view/cust-service-sim/",
            "remote_support": "Yes",
            "adaptive_support": "No",
            "test_type": ["Simulations"],
            "description": "Interactive customer service scenario simulation.",
            "duration": 45,
        },
        {
            "name": "SQL Server",
            "url": "https://www.shl.com/products/product-catalog/view/sql-server/",
            "remote_support": "Yes",
            "adaptive_support": "No",
            "test_type": ["Knowledge & Skills"],
            "description": "SQL Server queries, stored procedures, indexing.",
            "duration": 30,
        },
    ]


@pytest.fixture(scope="module")
def setup_search(sample_catalogue, tmp_path_factory):
    """Set up catalogue and embeddings, patch the search module."""
    import engine.search as search_mod
    from engine.embeddings import embed_query

    tmp_dir = tmp_path_factory.mktemp("search_test")
    cat_path = tmp_dir / "catalogue.json"
    emb_path = tmp_dir / "embeddings.npy"

    with open(cat_path, "w") as f:
        json.dump(sample_catalogue, f)

    embeddings = create_embeddings(sample_catalogue, show_progress=False)
    np.save(emb_path, embeddings)

    # Directly set module globals
    search_mod._catalogue = sample_catalogue
    search_mod._embeddings = embeddings

    return search_mod


class TestVectorSearch:
    def test_returns_correct_count(self, setup_search):
        results = setup_search.vector_search("test", top_k=3)
        assert len(results) == 3

    def test_sorted_by_score(self, setup_search):
        results = setup_search.vector_search("Python", top_k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_python_query_returns_python(self, setup_search):
        results = setup_search.vector_search("Python programming test", top_k=3)
        assert results[0]["name"] == "Python (New)"

    def test_java_query_returns_java(self, setup_search):
        results = setup_search.vector_search("Java developer assessment", top_k=3)
        assert results[0]["name"] == "Java 8"

    def test_sql_query_returns_sql(self, setup_search):
        results = setup_search.vector_search("SQL database assessment", top_k=3)
        assert results[0]["name"] == "SQL Server"

    def test_personality_query(self, setup_search):
        results = setup_search.vector_search("personality questionnaire", top_k=3)
        names = [r["name"] for r in results]
        assert "OPQ32r Personality" in names

    def test_results_have_score(self, setup_search):
        results = setup_search.vector_search("test", top_k=1)
        assert "score" in results[0]
        assert isinstance(results[0]["score"], float)

    def test_results_have_all_fields(self, setup_search):
        results = setup_search.vector_search("test", top_k=1)
        r = results[0]
        assert "name" in r
        assert "url" in r
        assert "test_type" in r
        assert isinstance(r["test_type"], list)  # array, not string!

    def test_top_k_capped_at_total(self, setup_search):
        results = setup_search.vector_search("test", top_k=100)
        assert len(results) == 6  # only 6 in catalogue
