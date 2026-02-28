"""
Embedding engine for SHL assessments.

Uses sentence-transformers with 'all-MiniLM-L6-v2' to create 384-dimensional
embeddings for each assessment. Runs locally on CPU — no API key needed.

Embedding text format (per the updated spec):
    f"{name}. Test type: {', '.join(test_type)}. {description}"

Including test_type in the text helps the model understand what KIND of
assessment it is, not just the topic.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
SCRAPER_DIR = Path(__file__).parent.parent / "scraper"
EMBEDDINGS_FILE = SCRAPER_DIR / "embeddings.npy"
CATALOGUE_FILE = SCRAPER_DIR / "catalogue.json"

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

    Format: "{name}. Test type: {test_type_list}. {description}"
    """
    name = assessment.get("name", "")
    test_type = assessment.get("test_type", [])
    description = assessment.get("description", "")

    parts = [name]

    if test_type:
        type_str = ", ".join(test_type) if isinstance(test_type, list) else str(test_type)
        parts.append(f"Test type: {type_str}")

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
        normalize_embeddings=True,  # Pre-normalize for cosine sim via dot product
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


def load_embeddings(filepath: Path = None) -> tuple:
    """
    Load catalogue + embeddings from disk.

    Returns:
        Tuple of (catalogue_list, embeddings_matrix)
    """
    if filepath is None:
        filepath = EMBEDDINGS_FILE
    cat_path = filepath.parent / "catalogue.json"

    if not filepath.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {filepath}. "
            "Run `python -m engine.embeddings` to generate them."
        )

    with open(cat_path, "r", encoding="utf-8") as f:
        catalogue = json.load(f)

    embeddings = np.load(filepath)
    logger.info(f"Loaded {len(catalogue)} assessments with embeddings shape {embeddings.shape}")
    return catalogue, embeddings


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.

    Returns:
        1D numpy array of shape (384,), L2-normalized.
    """
    model = get_model()
    embedding = model.encode(query, normalize_embeddings=True)
    return np.array(embedding, dtype=np.float32)


def build_and_save(catalogue_path: Path = None, embeddings_path: Path = None) -> tuple:
    """
    Full pipeline: load catalogue → create embeddings → save.

    Returns:
        Tuple of (assessments_list, embeddings_array).
    """
    if catalogue_path is None:
        catalogue_path = CATALOGUE_FILE
    if embeddings_path is None:
        embeddings_path = EMBEDDINGS_FILE

    with open(catalogue_path, "r", encoding="utf-8") as f:
        assessments = json.load(f)

    logger.info(f"Loaded {len(assessments)} assessments from {catalogue_path}")

    embeddings = create_embeddings(assessments)
    save_embeddings(embeddings, embeddings_path)

    return assessments, embeddings


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    assessments, embeddings = build_and_save()
    print(f"\nDone! Created {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
