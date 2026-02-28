"""
Tests for the vector search engine.

Tests cover:
  - Search returns correct number of results
  - Results are sorted by score (descending)
  - Similar queries return relevant results
  - Search with pre-computed embeddings
  - Engine initialization and properties
"""

import json
import pytest
import numpy as np
from pathlib import Path

from engine.embeddings import create_embeddings, save_embeddings
from engine.search import VectorSearchEngine


@pytest.fixture(scope="module")
def sample_catalogue():
    """Create a small test catalogue with diverse assessments."""
    return [
        {
            "name": "Python (New)",
            "url": "https://www.shl.com/products/product-catalog/view/python-new/",
            "remote_support": "Yes",
            "adaptive_irt": "No",
            "test_type_codes": ["K"],
            "test_types": ["Knowledge & Skills"],
            "test_type": "Knowledge & Skills",
            "description": "The Python test measures knowledge of programming in Python. Covers OOP, data structures, functions, and modules.",
            "duration": 30,
        },
        {
            "name": "Java 8",
            "url": "https://www.shl.com/products/product-catalog/view/java-8/",
            "remote_support": "Yes",
            "adaptive_irt": "No",
            "test_type_codes": ["K"],
            "test_types": ["Knowledge & Skills"],
            "test_type": "Knowledge & Skills",
            "description": "The Java 8 test evaluates knowledge of Java programming language including streams, lambda expressions, and concurrency.",
            "duration": 35,
        },
        {
            "name": "Verify G+ Cognitive Ability",
            "url": "https://www.shl.com/products/product-catalog/view/verify-g-plus/",
            "remote_support": "Yes",
            "adaptive_irt": "Yes",
            "test_type_codes": ["A"],
            "test_types": ["Ability & Aptitude"],
            "test_type": "Ability & Aptitude",
            "description": "Verify G+ is a cognitive ability assessment measuring general mental ability through numerical, verbal, and inductive reasoning.",
            "duration": 36,
        },
        {
            "name": "OPQ32r Personality Questionnaire",
            "url": "https://www.shl.com/products/product-catalog/view/opq32r/",
            "remote_support": "Yes",
            "adaptive_irt": "No",
            "test_type_codes": ["P"],
            "test_types": ["Personality & Behavior"],
            "test_type": "Personality & Behavior",
            "description": "The OPQ32r measures 32 personality characteristics relevant to the world of work, covering relationships, thinking style, and feelings.",
            "duration": 25,
        },
        {
            "name": "Customer Service Simulation",
            "url": "https://www.shl.com/products/product-catalog/view/customer-service-sim/",
            "remote_support": "Yes",
            "adaptive_irt": "No",
            "test_type_codes": ["S"],
            "test_types": ["Simulations"],
            "test_type": "Simulations",
            "description": "Interactive simulation of customer service scenarios including handling complaints, resolving issues, and maintaining satisfaction.",
            "duration": 45,
        },
        {
            "name": "SQL Server",
            "url": "https://www.shl.com/products/product-catalog/view/sql-server/",
            "remote_support": "Yes",
            "adaptive_irt": "No",
            "test_type_codes": ["K"],
            "test_types": ["Knowledge & Skills"],
            "test_type": "Knowledge & Skills",
            "description": "Tests knowledge of Microsoft SQL Server including queries, stored procedures, indexing, and database design.",
            "duration": 30,
        },
        {
            "name": "Graduate Manager Solution",
            "url": "https://www.shl.com/products/product-catalog/view/grad-manager/",
            "remote_support": "No",
            "adaptive_irt": "No",
            "test_type_codes": ["C", "P", "A"],
            "test_types": ["Competencies", "Personality & Behavior", "Ability & Aptitude"],
            "test_type": "Competencies, Personality & Behavior, Ability & Aptitude",
            "description": "Comprehensive assessment solution for graduate manager candidates combining competency evaluation, personality profiling, and ability testing.",
            "duration": 60,
        },
    ]


@pytest.fixture(scope="module")
def search_engine(sample_catalogue, tmp_path_factory):
    """Create a search engine with the sample catalogue."""
    tmp_dir = tmp_path_factory.mktemp("search_test")

    # Save catalogue
    catalogue_path = tmp_dir / "catalogue.json"
    with open(catalogue_path, "w") as f:
        json.dump(sample_catalogue, f)

    # Create and save embeddings
    embeddings = create_embeddings(sample_catalogue, show_progress=False)
    embeddings_path = tmp_dir / "embeddings.npy"
    save_embeddings(embeddings, embeddings_path)

    # Create engine
    return VectorSearchEngine(
        catalogue_path=catalogue_path,
        embeddings_path=embeddings_path,
    )


class TestVectorSearchEngine:
    """Tests for the VectorSearchEngine class."""

    def test_engine_size(self, search_engine):
        assert search_engine.size == 7

    def test_search_returns_correct_count(self, search_engine):
        results = search_engine.search("programming", top_k=3)
        assert len(results) == 3

    def test_search_top_k_capped(self, search_engine):
        # Requesting more than available should return all
        results = search_engine.search("test", top_k=100)
        assert len(results) == 7

    def test_results_sorted_by_score(self, search_engine):
        results = search_engine.search("Python developer", top_k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), "Results should be sorted by descending score"

    def test_results_have_score_field(self, search_engine):
        results = search_engine.search("test", top_k=3)
        for r in results:
            assert "score" in r
            assert isinstance(r["score"], float)
            assert -1.0 <= r["score"] <= 1.0

    def test_python_query_returns_python_first(self, search_engine):
        results = search_engine.search("Python programming test", top_k=3)
        # Python (New) should be the top result
        assert results[0]["name"] == "Python (New)", (
            f"Expected 'Python (New)' as top result, got '{results[0]['name']}'"
        )

    def test_java_query_returns_java_first(self, search_engine):
        results = search_engine.search("Java developer assessment", top_k=3)
        assert results[0]["name"] == "Java 8", (
            f"Expected 'Java 8' as top result, got '{results[0]['name']}'"
        )

    def test_personality_query_returns_personality(self, search_engine):
        results = search_engine.search("personality questionnaire for workplace", top_k=3)
        # OPQ32r should be in top results
        top_names = [r["name"] for r in results]
        assert "OPQ32r Personality Questionnaire" in top_names, (
            f"Expected OPQ32r in top 3, got {top_names}"
        )

    def test_cognitive_query_returns_cognitive(self, search_engine):
        results = search_engine.search("cognitive ability reasoning", top_k=3)
        top_names = [r["name"] for r in results]
        assert "Verify G+ Cognitive Ability" in top_names, (
            f"Expected Verify G+ in top 3, got {top_names}"
        )

    def test_sql_query_returns_sql(self, search_engine):
        results = search_engine.search("SQL database assessment", top_k=3)
        assert results[0]["name"] == "SQL Server", (
            f"Expected 'SQL Server' as top result, got '{results[0]['name']}'"
        )

    def test_results_contain_all_fields(self, search_engine):
        results = search_engine.search("any test", top_k=1)
        assert len(results) == 1
        r = results[0]
        assert "name" in r
        assert "url" in r
        assert "remote_support" in r
        assert "adaptive_irt" in r
        assert "test_type" in r
        assert "duration" in r
        assert "score" in r

    def test_get_assessment_by_name(self, search_engine):
        result = search_engine.get_assessment_by_name("Python (New)")
        assert result is not None
        assert result["name"] == "Python (New)"
        assert result["duration"] == 30

    def test_get_assessment_by_name_not_found(self, search_engine):
        result = search_engine.get_assessment_by_name("Nonexistent Test")
        assert result is None

    def test_get_all_test_types(self, search_engine):
        types = search_engine.get_all_test_types()
        assert "Knowledge & Skills" in types
        assert "Ability & Aptitude" in types
        assert "Personality & Behavior" in types
        assert "Simulations" in types

    def test_search_with_embedding(self, search_engine):
        from engine.embeddings import embed_query
        query_emb = embed_query("Python programming")
        results = search_engine.search_with_embedding(query_emb, top_k=3)
        assert len(results) == 3
        assert results[0]["name"] == "Python (New)"
