"""
Tests for the recommendation pipeline.

Tests cover:
  - Full pipeline integration (query → parse → search → filter → format)
  - Filter behavior (duration, remote, test type boost)
  - Output format validation
  - Edge cases
"""

import json
import pytest
import numpy as np
from pathlib import Path

from engine.embeddings import create_embeddings, save_embeddings
from engine.search import VectorSearchEngine
from engine.recommender import recommend, apply_filters, format_results


@pytest.fixture(scope="module")
def sample_catalogue():
    """Diverse test catalogue for recommender testing."""
    return [
        {
            "name": "Python (New)",
            "url": "https://www.shl.com/products/product-catalog/view/python-new/",
            "remote_support": "Yes",
            "adaptive_irt": "No",
            "test_type_codes": ["K"],
            "test_types": ["Knowledge & Skills"],
            "test_type": "Knowledge & Skills",
            "description": "Tests Python programming knowledge including OOP, data structures, and modules.",
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
            "description": "Tests Java 8 features including streams, lambda, and concurrency.",
            "duration": 35,
        },
        {
            "name": "JavaScript (New)",
            "url": "https://www.shl.com/products/product-catalog/view/javascript-new/",
            "remote_support": "Yes",
            "adaptive_irt": "No",
            "test_type_codes": ["K"],
            "test_types": ["Knowledge & Skills"],
            "test_type": "Knowledge & Skills",
            "description": "Tests JavaScript programming including ES6+, DOM, and async programming.",
            "duration": 30,
        },
        {
            "name": "Verify G+ Cognitive",
            "url": "https://www.shl.com/products/product-catalog/view/verify-g-plus/",
            "remote_support": "Yes",
            "adaptive_irt": "Yes",
            "test_type_codes": ["A"],
            "test_types": ["Ability & Aptitude"],
            "test_type": "Ability & Aptitude",
            "description": "Cognitive ability assessment measuring numerical, verbal, and inductive reasoning.",
            "duration": 36,
        },
        {
            "name": "OPQ32r Personality",
            "url": "https://www.shl.com/products/product-catalog/view/opq32r/",
            "remote_support": "Yes",
            "adaptive_irt": "No",
            "test_type_codes": ["P"],
            "test_types": ["Personality & Behavior"],
            "test_type": "Personality & Behavior",
            "description": "Measures 32 personality characteristics for workplace assessment.",
            "duration": 25,
        },
        {
            "name": "Graduate Manager Solution",
            "url": "https://www.shl.com/products/product-catalog/view/grad-manager/",
            "remote_support": "No",
            "adaptive_irt": "No",
            "test_type_codes": ["C", "P", "A"],
            "test_types": ["Competencies", "Personality & Behavior", "Ability & Aptitude"],
            "test_type": "Competencies, Personality & Behavior, Ability & Aptitude",
            "description": "Comprehensive assessment for graduate manager candidates.",
            "duration": 90,
        },
        {
            "name": "Customer Service Sim",
            "url": "https://www.shl.com/products/product-catalog/view/cust-service-sim/",
            "remote_support": "Yes",
            "adaptive_irt": "No",
            "test_type_codes": ["S"],
            "test_types": ["Simulations"],
            "test_type": "Simulations",
            "description": "Interactive simulation of customer service scenarios.",
            "duration": 45,
        },
    ]


@pytest.fixture(scope="module")
def test_engine(sample_catalogue, tmp_path_factory):
    """Search engine initialized with sample data."""
    tmp_dir = tmp_path_factory.mktemp("recommender_test")

    catalogue_path = tmp_dir / "catalogue.json"
    with open(catalogue_path, "w") as f:
        json.dump(sample_catalogue, f)

    embeddings = create_embeddings(sample_catalogue, show_progress=False)
    embeddings_path = tmp_dir / "embeddings.npy"
    save_embeddings(embeddings, embeddings_path)

    return VectorSearchEngine(
        catalogue_path=catalogue_path,
        embeddings_path=embeddings_path,
    )


# --- Filter Tests ---

class TestApplyFilters:
    """Tests for the filtering and re-ranking logic."""

    def _make_candidate(self, name="Test", duration=30, remote="Yes",
                        test_types=None, score=0.5):
        return {
            "name": name,
            "duration": duration,
            "remote_support": remote,
            "test_types": test_types or [],
            "score": score,
        }

    def test_duration_filter(self):
        candidates = [
            self._make_candidate("Short", duration=20, score=0.8),
            self._make_candidate("Long", duration=60, score=0.9),
            self._make_candidate("Medium", duration=30, score=0.7),
        ]
        constraints = {"max_duration": 30, "test_types": [], "remote_required": None}
        results = apply_filters(candidates, constraints)

        names = [r["name"] for r in results]
        assert "Long" not in names
        assert "Short" in names
        assert "Medium" in names

    def test_duration_filter_with_null(self):
        """Assessments with null duration should NOT be filtered out."""
        candidates = [
            self._make_candidate("No Duration", duration=None, score=0.8),
            self._make_candidate("Has Duration", duration=60, score=0.9),
        ]
        constraints = {"max_duration": 30, "test_types": [], "remote_required": None}
        results = apply_filters(candidates, constraints)

        names = [r["name"] for r in results]
        assert "No Duration" in names  # null duration → keep
        assert "Has Duration" not in names  # 60 > 30 → remove

    def test_remote_filter(self):
        candidates = [
            self._make_candidate("Remote", remote="Yes", score=0.8),
            self._make_candidate("In-Person", remote="No", score=0.9),
        ]
        constraints = {"max_duration": None, "test_types": [], "remote_required": True}
        results = apply_filters(candidates, constraints)

        names = [r["name"] for r in results]
        assert "Remote" in names
        assert "In-Person" not in names

    def test_test_type_boost(self):
        candidates = [
            self._make_candidate("Wrong Type", test_types=["Simulations"], score=0.8),
            self._make_candidate("Right Type", test_types=["Knowledge & Skills"], score=0.7),
        ]
        constraints = {
            "max_duration": None,
            "test_types": ["Knowledge & Skills"],
            "remote_required": None,
        }
        results = apply_filters(candidates, constraints)

        # Right Type (0.7 + 0.3 = 1.0) should outrank Wrong Type (0.8)
        assert results[0]["name"] == "Right Type"
        assert results[0]["boosted_score"] == pytest.approx(1.0)

    def test_no_constraints(self):
        candidates = [
            self._make_candidate(score=0.8),
            self._make_candidate(score=0.9),
        ]
        constraints = {"max_duration": None, "test_types": [], "remote_required": None}
        results = apply_filters(candidates, constraints)
        assert len(results) == 2

    def test_combined_filters(self):
        candidates = [
            self._make_candidate("A", duration=20, remote="Yes", test_types=["Knowledge & Skills"], score=0.7),
            self._make_candidate("B", duration=60, remote="Yes", test_types=["Knowledge & Skills"], score=0.9),
            self._make_candidate("C", duration=20, remote="No", test_types=["Knowledge & Skills"], score=0.85),
            self._make_candidate("D", duration=20, remote="Yes", test_types=["Simulations"], score=0.8),
        ]
        constraints = {
            "max_duration": 30,
            "test_types": ["Knowledge & Skills"],
            "remote_required": True,
        }
        results = apply_filters(candidates, constraints)

        names = [r["name"] for r in results]
        assert "B" not in names  # too long
        assert "C" not in names  # not remote
        assert "A" in names
        assert "D" in names


# --- Output Format Tests ---

class TestFormatResults:
    """Tests for output formatting."""

    def test_format_has_required_fields(self):
        results = [{
            "name": "Python (New)",
            "url": "https://shl.com/python",
            "remote_support": "Yes",
            "adaptive_irt": "No",
            "duration": 30,
            "test_type": "Knowledge & Skills",
        }]
        formatted = format_results(results)

        assert len(formatted) == 1
        r = formatted[0]
        assert r["assessment_name"] == "Python (New)"
        assert r["url"] == "https://shl.com/python"
        assert r["remote_support"] == "Yes"
        assert r["adaptive_irt"] == "No"
        assert r["duration"] == 30
        assert r["test_type"] == "Knowledge & Skills"

    def test_format_null_duration(self):
        results = [{"name": "Test", "url": "http://x", "remote_support": "No",
                     "adaptive_irt": "No", "duration": None, "test_type": "Unknown"}]
        formatted = format_results(results)
        assert formatted[0]["duration"] is None

    def test_format_exactly_six_fields(self):
        results = [{"name": "Test", "url": "http://x", "remote_support": "Yes",
                     "adaptive_irt": "No", "duration": 30, "test_type": "K",
                     "extra_field": "should be excluded"}]
        formatted = format_results(results)
        assert set(formatted[0].keys()) == {
            "assessment_name", "url", "remote_support",
            "adaptive_irt", "duration", "test_type"
        }


# --- Integration Tests ---

class TestRecommendPipeline:
    """Integration tests for the full recommendation pipeline."""

    def test_python_recommendation(self, test_engine):
        results = recommend("Python programming test", top_k=3, engine=test_engine)
        assert len(results) <= 3
        assert results[0]["assessment_name"] == "Python (New)"

    def test_recommendation_format(self, test_engine):
        results = recommend("any test", top_k=5, engine=test_engine)
        for r in results:
            assert "assessment_name" in r
            assert "url" in r
            assert "remote_support" in r
            assert "adaptive_irt" in r
            assert "duration" in r
            assert "test_type" in r

    def test_duration_constraint(self, test_engine):
        results = recommend(
            "programming assessment under 30 minutes",
            top_k=10,
            engine=test_engine,
        )
        for r in results:
            if r["duration"] is not None:
                assert r["duration"] <= 30, (
                    f"Assessment '{r['assessment_name']}' has duration {r['duration']} > 30"
                )

    def test_remote_constraint(self, test_engine):
        results = recommend(
            "online remote assessment",
            top_k=10,
            engine=test_engine,
        )
        for r in results:
            assert r["remote_support"] == "Yes", (
                f"Assessment '{r['assessment_name']}' is not remote"
            )

    def test_returns_max_top_k(self, test_engine):
        results = recommend("test", top_k=3, engine=test_engine)
        assert len(results) <= 3

    def test_empty_query_handling(self, test_engine):
        # Should not crash, may return results based on empty embedding
        results = recommend("", top_k=5, engine=test_engine)
        assert isinstance(results, list)
