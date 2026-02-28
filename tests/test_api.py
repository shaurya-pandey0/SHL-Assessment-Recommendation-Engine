"""
Tests for the FastAPI endpoints.

Tests:
  - GET /health returns 200 and correct format
  - POST /recommend returns valid response format
  - POST /recommend with empty query returns 400
  - Response fields match SHL spec exactly
"""

import json
import pytest
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient

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
            "description": "Tests Python programming.",
            "duration": 30,
        },
        {
            "name": "Java 8",
            "url": "https://www.shl.com/products/product-catalog/view/java-8/",
            "remote_support": "Yes",
            "adaptive_support": "No",
            "test_type": ["Knowledge & Skills"],
            "description": "Tests Java.",
            "duration": 35,
        },
        {
            "name": "OPQ32r",
            "url": "https://www.shl.com/products/product-catalog/view/opq32r/",
            "remote_support": "Yes",
            "adaptive_support": "No",
            "test_type": ["Personality & Behavior"],
            "description": "Personality assessment.",
            "duration": 25,
        },
    ]


@pytest.fixture(scope="module")
def client(sample_catalogue):
    """Create test client with mock data."""
    import engine.search as search_mod

    embeddings = create_embeddings(sample_catalogue, show_progress=False)
    search_mod._catalogue = sample_catalogue
    search_mod._embeddings = embeddings

    from api.main import app
    return TestClient(app)


REQUIRED_FIELDS = {"url", "name", "adaptive_support", "description",
                   "duration", "remote_support", "test_type"}


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_healthy(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"


class TestRecommendEndpoint:
    def test_recommend_returns_200(self, client):
        response = client.post("/recommend", json={"query": "Python test"})
        assert response.status_code == 200

    def test_recommend_has_recommended_assessments(self, client):
        response = client.post("/recommend", json={"query": "Python"})
        data = response.json()
        assert "recommended_assessments" in data

    def test_recommend_returns_results(self, client):
        response = client.post("/recommend", json={"query": "Python"})
        data = response.json()
        assert len(data["recommended_assessments"]) > 0

    def test_recommend_result_fields(self, client):
        response = client.post("/recommend", json={"query": "Python"})
        data = response.json()
        for r in data["recommended_assessments"]:
            assert set(r.keys()) == REQUIRED_FIELDS

    def test_recommend_test_type_is_array(self, client):
        response = client.post("/recommend", json={"query": "test"})
        data = response.json()
        for r in data["recommended_assessments"]:
            assert isinstance(r["test_type"], list)

    def test_recommend_empty_query_returns_400(self, client):
        response = client.post("/recommend", json={"query": ""})
        assert response.status_code == 400

    def test_recommend_max_10_results(self, client):
        response = client.post("/recommend", json={"query": "test"})
        data = response.json()
        assert len(data["recommended_assessments"]) <= 10
