"""
Embedding engine for SHL assessments.

Uses sentence-transformers with 'all-MiniLM-L6-v2' to create 384-dimensional
embeddings for each assessment. The model runs locally on CPU — no API key needed.

Embedding text is composed from:
    f"{name}. {test_type}. {description}"

This captures the assessment's identity, category, and detailed functionality
in a single semantic vector.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
DATA_DIR = Path(__file__).parent.parent / "data"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npy"
CATALOGUE_FILE = DATA_DIR / "catalogue.json"

# Lazy-loaded model singleton
_model = None


def get_model():
    """Load the sentence-transformer model (lazy singleton)."""
    global _model
    if _model is None:
        logger.info(f"Loading sentence-transformer model: {MODEL_NAME}...")
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
        logger.info("Model loaded successfully.")
    return _model


def build_embedding_text(assessment: dict) -> str:
    """
    Build the text to embed for a single assessment.

    Combines name, test type, and description to capture
    both identity and functionality.
    """
    name = assessment.get("name", "")
    test_type = assessment.get("test_type", "")
    description = assessment.get("description", "")

    parts = [name]
    if test_type and test_type != "Unknown":
        parts.append(test_type)
    if description:
        parts.append(description)

    return ". ".join(parts)


def create_embeddings(
    assessments: list[dict],
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Create embeddings for a list of assessments.

    Args:
        assessments: List of assessment dicts from the catalogue.
        batch_size: Batch size for encoding.
        show_progress: Whether to show a progress bar.

    Returns:
        numpy array of shape (n_assessments, 384).
    """
    model = get_model()

    texts = [build_embedding_text(a) for a in assessments]
    logger.info(f"Creating embeddings for {len(texts)} assessments...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,  # Pre-normalize for cosine similarity
    )

    embeddings = np.array(embeddings, dtype=np.float32)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    return embeddings


def save_embeddings(embeddings: np.ndarray, filepath: Path = None) -> Path:
    """Save embeddings to a .npy file."""
    if filepath is None:
        filepath = EMBEDDINGS_FILE

    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.save(filepath, embeddings)
    logger.info(f"Saved embeddings to {filepath}")
    return filepath


def load_embeddings(filepath: Path = None) -> np.ndarray:
    """Load embeddings from a .npy file."""
    if filepath is None:
        filepath = EMBEDDINGS_FILE

    if not filepath.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {filepath}. "
            "Run `python -m engine.embeddings` to generate them."
        )

    embeddings = np.load(filepath)
    logger.info(f"Loaded embeddings with shape {embeddings.shape}")
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.

    Returns:
        1D numpy array of shape (384,).
    """
    model = get_model()
    embedding = model.encode(
        query,
        normalize_embeddings=True,
    )
    return np.array(embedding, dtype=np.float32)


def build_and_save(catalogue_path: Path = None, embeddings_path: Path = None) -> tuple:
    """
    Full pipeline: load catalogue → create embeddings → save to disk.

    Returns:
        Tuple of (assessments list, embeddings array).
    """
    if catalogue_path is None:
        catalogue_path = CATALOGUE_FILE
    if embeddings_path is None:
        embeddings_path = EMBEDDINGS_FILE

    # Load catalogue
    with open(catalogue_path, "r", encoding="utf-8") as f:
        assessments = json.load(f)

    logger.info(f"Loaded {len(assessments)} assessments from {catalogue_path}")

    # Create and save embeddings
    embeddings = create_embeddings(assessments)
    save_embeddings(embeddings, embeddings_path)

    return assessments, embeddings


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    assessments, embeddings = build_and_save()
    print(f"\nDone! Created {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
