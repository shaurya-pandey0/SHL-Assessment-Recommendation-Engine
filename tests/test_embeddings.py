"""
Tests for the embedding engine.

Tests:
  - Embedding text construction (includes test_type array)
  - Embedding shape and normalization
  - Save/load round-trip
  - Semantic similarity sanity checks
"""

import pytest
import numpy as np
from pathlib import Path

from engine.embeddings import (
    build_embedding_text,
    create_embeddings,
    save_embeddings,
    load_embeddings,
    embed_query,
)


class TestBuildEmbeddingText:
    def test_full_assessment(self):
        a = {
            "name": "Python (New)",
            "test_type": ["Knowledge & Skills"],
            "description": "Tests Python programming knowledge.",
        }
        text = build_embedding_text(a)
        assert "Python (New)" in text
        assert "Knowledge & Skills" in text
        assert "Python programming" in text

    def test_multiple_test_types(self):
        a = {
            "name": "Combo Test",
            "test_type": ["Knowledge & Skills", "Personality & Behavior"],
            "description": "A combo.",
        }
        text = build_embedding_text(a)
        assert "Knowledge & Skills" in text
        assert "Personality & Behavior" in text

    def test_no_description(self):
        a = {"name": "Test", "test_type": ["Cognitive"], "description": ""}
        text = build_embedding_text(a)
        assert "Test" in text
        assert "Cognitive" in text

    def test_empty_test_type(self):
        a = {"name": "Test", "test_type": [], "description": "Some desc"}
        text = build_embedding_text(a)
        assert "Test" in text
        assert "Test type" not in text  # empty array → no type text

    def test_empty_assessment(self):
        text = build_embedding_text({})
        assert isinstance(text, str)


class TestEmbeddings:
    @pytest.fixture
    def sample_assessments(self):
        return [
            {"name": "Python (New)", "test_type": ["Knowledge & Skills"], "description": "Tests Python."},
            {"name": "Java 8", "test_type": ["Knowledge & Skills"], "description": "Tests Java."},
            {"name": "Leadership Assessment", "test_type": ["Personality & Behavior"], "description": "Leadership."},
        ]

    def test_shape(self, sample_assessments):
        emb = create_embeddings(sample_assessments, show_progress=False)
        assert emb.shape == (3, 384)
        assert emb.dtype == np.float32

    def test_normalized(self, sample_assessments):
        emb = create_embeddings(sample_assessments, show_progress=False)
        norms = np.linalg.norm(emb, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_similar_more_similar(self, sample_assessments):
        emb = create_embeddings(sample_assessments, show_progress=False)
        py_java = np.dot(emb[0], emb[1])
        py_lead = np.dot(emb[0], emb[2])
        assert py_java > py_lead

    def test_save_load(self, sample_assessments, tmp_path):
        import json
        emb = create_embeddings(sample_assessments, show_progress=False)
        emb_path = tmp_path / "embeddings.npy"
        cat_path = tmp_path / "catalogue.json"
        np.save(emb_path, emb)
        with open(cat_path, "w") as f:
            json.dump(sample_assessments, f)
        cat, loaded = load_embeddings(emb_path)
        np.testing.assert_array_equal(emb, loaded)
        assert len(cat) == 3


class TestEmbedQuery:
    def test_shape(self):
        emb = embed_query("Python developer assessment")
        assert emb.shape == (384,)

    def test_normalized(self):
        emb = embed_query("test query")
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-5

    def test_similar_queries(self):
        e1 = embed_query("Python programming test")
        e2 = embed_query("Python coding assessment")
        e3 = embed_query("cooking recipe book")
        assert np.dot(e1, e2) > np.dot(e1, e3)
