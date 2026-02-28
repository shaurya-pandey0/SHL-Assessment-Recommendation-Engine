"""
Tests for the embedding engine.

Tests cover:
  - Embedding text construction
  - Embedding shape and normalization
  - Save/load round-trip
  - Query embedding
"""

import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from engine.embeddings import (
    build_embedding_text,
    create_embeddings,
    save_embeddings,
    load_embeddings,
    embed_query,
)


class TestBuildEmbeddingText:
    """Tests for the embedding text construction function."""

    def test_full_assessment(self):
        assessment = {
            "name": "Python (New)",
            "test_type": "Knowledge & Skills",
            "description": "Tests Python programming knowledge including OOP and data structures.",
        }
        text = build_embedding_text(assessment)
        assert "Python (New)" in text
        assert "Knowledge & Skills" in text
        assert "Python programming" in text

    def test_no_description(self):
        assessment = {
            "name": "Test Assessment",
            "test_type": "Cognitive",
            "description": "",
        }
        text = build_embedding_text(assessment)
        assert "Test Assessment" in text
        assert "Cognitive" in text
        # Should not have trailing period/space issues
        assert text.strip() == text

    def test_unknown_test_type(self):
        assessment = {
            "name": "Mystery Test",
            "test_type": "Unknown",
            "description": "Some description",
        }
        text = build_embedding_text(assessment)
        assert "Mystery Test" in text
        assert "Unknown" not in text  # Unknown should be excluded
        assert "Some description" in text

    def test_empty_assessment(self):
        assessment = {}
        text = build_embedding_text(assessment)
        assert isinstance(text, str)

    def test_missing_fields(self):
        assessment = {"name": "Just a Name"}
        text = build_embedding_text(assessment)
        assert text == "Just a Name"


class TestEmbeddings:
    """Tests for embedding creation, saving, and loading."""

    @pytest.fixture
    def sample_assessments(self):
        return [
            {
                "name": "Python (New)",
                "test_type": "Knowledge & Skills",
                "description": "Tests Python programming.",
            },
            {
                "name": "Java 8",
                "test_type": "Knowledge & Skills",
                "description": "Tests Java 8 features.",
            },
            {
                "name": "Leadership Assessment",
                "test_type": "Personality & Behavior",
                "description": "Evaluates leadership qualities.",
            },
        ]

    def test_create_embeddings_shape(self, sample_assessments):
        embeddings = create_embeddings(sample_assessments, show_progress=False)
        assert embeddings.shape == (3, 384)
        assert embeddings.dtype == np.float32

    def test_embeddings_are_normalized(self, sample_assessments):
        embeddings = create_embeddings(sample_assessments, show_progress=False)
        # Check L2 norms are approximately 1.0
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_similar_assessments_have_similar_embeddings(self, sample_assessments):
        embeddings = create_embeddings(sample_assessments, show_progress=False)

        # Python and Java should be more similar than Python and Leadership
        python_java_sim = np.dot(embeddings[0], embeddings[1])
        python_leadership_sim = np.dot(embeddings[0], embeddings[2])

        assert python_java_sim > python_leadership_sim, (
            f"Python-Java similarity ({python_java_sim:.3f}) should be greater "
            f"than Python-Leadership similarity ({python_leadership_sim:.3f})"
        )

    def test_save_and_load_round_trip(self, sample_assessments, tmp_path):
        embeddings = create_embeddings(sample_assessments, show_progress=False)

        filepath = tmp_path / "test_embeddings.npy"
        save_embeddings(embeddings, filepath)
        loaded = load_embeddings(filepath)

        np.testing.assert_array_equal(embeddings, loaded)

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_embeddings(tmp_path / "nonexistent.npy")


class TestEmbedQuery:
    """Tests for query embedding."""

    def test_query_embedding_shape(self):
        embedding = embed_query("Python developer assessment")
        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32

    def test_query_embedding_normalized(self):
        embedding = embed_query("test query")
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-5

    def test_similar_queries_have_similar_embeddings(self):
        emb1 = embed_query("Python programming test")
        emb2 = embed_query("Python coding assessment")
        emb3 = embed_query("cooking recipe book")

        sim_related = np.dot(emb1, emb2)
        sim_unrelated = np.dot(emb1, emb3)

        assert sim_related > sim_unrelated, (
            f"Related queries similarity ({sim_related:.3f}) should be greater "
            f"than unrelated queries similarity ({sim_unrelated:.3f})"
        )
