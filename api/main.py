"""
SHL Assessment Recommendation API

Endpoints:
  GET  /health     → {"status": "healthy"}
  POST /recommend  → {"recommended_assessments": [...]}

Accepts both:
  - Text queries: {"query": "Python developer assessment"}
  - URL queries:  {"query": "https://example.com/job-posting"} → fetches & extracts text
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

import requests as req_lib
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engine.recommender import recommend

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load embeddings on startup for fast first request."""
    logger.info("Pre-loading embeddings...")
    try:
        from engine.search import _ensure_loaded
        _ensure_loaded()
        logger.info("Embeddings loaded. API ready.")
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
    yield


# ── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SHL Assessment Recommendation Engine",
    description="Recommends SHL assessments based on job descriptions or natural language queries.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str


class AssessmentResponse(BaseModel):
    url: str
    name: str
    adaptive_support: str
    description: str
    duration: Optional[int]
    remote_support: str
    test_type: list[str]


class RecommendResponse(BaseModel):
    recommended_assessments: list[AssessmentResponse]


# ── URL Text Extraction ─────────────────────────────────────────────────────

def extract_text_from_url(url: str) -> str:
    """Fetch a URL and extract meaningful text content."""
    try:
        resp = req_lib.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (compatible; SHLBot/1.0)"
        })
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # Remove script and style elements
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)

        # Truncate to reasonable length for embedding
        if len(text) > 5000:
            text = text[:5000]

        return text

    except Exception as e:
        logger.warning(f"Failed to extract text from URL {url}: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Could not fetch URL content: {str(e)}"
        )


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/recommend", response_model=RecommendResponse)
def get_recommendations(request: QueryRequest):
    """
    Get assessment recommendations.

    Accepts natural language queries or job description URLs.
    Returns up to 10 recommended SHL assessments.
    """
    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # If query looks like a URL, fetch and extract text
    if query.startswith("http://") or query.startswith("https://"):
        logger.info(f"URL input detected, fetching: {query[:100]}...")
        query = extract_text_from_url(query)
        if not query.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from URL")

    logger.info(f"Processing query: {query[:200]}...")

    try:
        results = recommend(query, top_k=10)
    except Exception as e:
        logger.error(f"Recommendation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal recommendation error")

    return {"recommended_assessments": results}
