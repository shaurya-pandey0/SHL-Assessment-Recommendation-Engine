"""
Embedding engine for SHL assessments.

Uses nomic-embed-text-v1.5 (GGUF quantized, Q8_0) via llama-cpp-python
to create 768-dimensional embeddings for each assessment. Runs locally
on CPU — no API key needed.

Nomic-embed-text-v1.5 is a strong general-purpose embedding model that
outperforms many larger models on retrieval benchmarks. The GGUF format
enables fast CPU inference with minimal memory footprint.

Embedding text format (enriched):
    f"search_document: {name}. Test type: {types}. {description}. {domain_hints}"

The "search_document:" prefix is required by nomic-embed for document embeddings.
Query embeddings use "search_query:" prefix instead.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

def _find_model_path() -> Path:
    """Find the nomic-embed GGUF model file (supports any quantization)."""
    root = Path(__file__).parent.parent
    candidates = sorted(root.glob("nomic-embed-text-v1.5*.gguf"))
    if candidates:
        return candidates[0]
    # Fallback to default name
    return root / "nomic-embed-text-v1.5.Q8_0.gguf"

MODEL_PATH = _find_model_path()
SCRAPER_DIR = Path(__file__).parent.parent / "scraper"
EMBEDDINGS_FILE = SCRAPER_DIR / "embeddings.npy"
CATALOGUE_FILE = SCRAPER_DIR / "catalogue.json"

# Lazy-loaded model singleton
_model = None


def get_model():
    """Load the nomic-embed GGUF model (lazy singleton)."""
    global _model
    if _model is None:
        logger.info(f"Loading nomic-embed-text-v1.5 from {MODEL_PATH}...")
        from llama_cpp import Llama
        _model = Llama(
            model_path=str(MODEL_PATH),
            embedding=True,
            verbose=False,
            n_ctx=2048,  # nomic-embed supports up to 2048 tokens
        )
        logger.info("Nomic-embed model loaded successfully.")
    return _model


def build_embedding_text(assessment: dict) -> str:
    """
    Build the text to embed for a single assessment.

    Format:
        "search_document: {name}. Test type: {test_type_list}. {description}. Keywords: {enrichment}"

    Enrichment appends a short set of domain-prior keywords per test type
    to nudge each assessment's embedding toward the right semantic cluster.
    """
    TEST_TYPE_ENRICHMENT = {
        "Knowledge & Skills":             "technical skills expertise proficiency tool",
        "Ability & Aptitude":             "cognitive reasoning analytical aptitude intelligence",
        "Personality & Behavior":         "personality behavior collaboration communication culture",
        "Simulations":                    "simulation practical scenario hands-on task",
        "Competencies":                   "competency leadership decision making behavioural",
        "Biodata & Situational Judgement":"situational judgement judgement workplace scenarios",
        "Development & 360":              "development feedback growth coaching learning",
        "Assessment Exercises":           "exercise role play in-tray written assessment",
    }

    name = assessment.get("name", "")
    test_type = assessment.get("test_type", [])
    description = assessment.get("description", "")

    parts = [name]

    if test_type:
        type_str = ", ".join(test_type) if isinstance(test_type, list) else str(test_type)
        parts.append(f"Test type: {type_str}")

    if description:
        parts.append(description)

    # Append short enrichment keywords per test type
    types = test_type if isinstance(test_type, list) else [test_type]
    hints = [TEST_TYPE_ENRICHMENT[t] for t in types if t in TEST_TYPE_ENRICHMENT]
    if hints:
        parts.append("Keywords: " + ". ".join(hints))

    # Nomic-embed requires "search_document:" prefix for documents
    return "search_document: " + ". ".join(parts)


def _normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize a vector or matrix of vectors."""
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return v / norms


def create_embeddings(
    assessments: list[dict],
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Create embeddings for a list of assessments.

    Returns:
        numpy array of shape (n_assessments, 768), L2-normalized.
    """
    model = get_model()

    texts = [build_embedding_text(a) for a in assessments]
    logger.info(f"Creating embeddings for {len(texts)} assessments...")

    all_embeddings = []
    for i, text in enumerate(texts):
        # Truncate to ~1500 chars to stay within 2048 token context window
        if len(text) > 1500:
            text = text[:1500]
        emb = model.embed(text)
        all_embeddings.append(emb)
        if show_progress and (i + 1) % 50 == 0:
            logger.info(f"  Embedded {i + 1}/{len(texts)}")

    embeddings = np.array(all_embeddings, dtype=np.float32)
    embeddings = _normalize(embeddings)
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
        1D numpy array of shape (768,), L2-normalized.
    """
    model = get_model()
    # Nomic-embed requires "search_query:" prefix for queries
    embedding = model.embed("search_query: " + query)
    embedding = np.array(embedding, dtype=np.float32)
    return _normalize(embedding)


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
