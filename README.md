# SHL Assessment Recommendation Engine

## Live Links

| Resource | URL |
|----------|-----|
| **API Endpoint** | `http://34.29.43.125:8000` |
| **API Docs (Swagger)** | `http://34.29.43.125:8000/docs` |
| **Frontend** | `http://35.225.189.150:8501/` |
| **GitHub** | [github.com/shaurya-pandey0/SHL-Assessment-Recommendation-Engine](https://github.com/shaurya-pandey0/SHL-Assessment-Recommendation-Engine) |

---

## Problem Statement

This system recommends relevant SHL assessments given a natural language query or job description URL. It uses a **hybrid retrieval pipeline**: semantic vector search (nomic-embed-text-v1.5, 768-dim) combined with keyword-based catalogue matching, structured constraint extraction (via Google Gemini with a rule-based fallback), and test-type-aware re-ranking. The system scrapes 389 individual test solutions from the SHL product catalogue and serves results through a FastAPI backend with a Streamlit frontend.

---

## Architecture

```
User Query (text or URL)
        │
        ▼
┌─────────────────────┐
│  URL Text Extractor  │  ← If input is a URL, fetch and extract text
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Query Parser        │  ← Gemini 1.5 Flash (+ rule-based fallback)
│  Extract: skills,    │     Extract structured intent:
│  duration, types     │     job_role, skills, duration, test_types
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  nomic-embed-text    │  ← Embed query into 768-dim vector
│  v1.5 (GGUF Q8_0)   │     Same model used for catalogue embeddings
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Hybrid Search       │  ← Vector search (top 50 candidates)
│  Vector + Keyword    │     + Keyword catalogue scan for exact matches
│  + Role-focused      │     + Role-focused vector search
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Re-ranker           │  ← Duration filter (hard constraint)
│  + Type Balancer     │     Type boost + keyword boost
│                      │     Smart balance (min 2 minority)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Top 10 Results      │  ← JSON via FastAPI
│  (formatted output)  │     Exact SHL field names
└─────────────────────┘
```

---

## Design Decisions

### Why nomic-embed-text-v1.5

Runs locally on CPU via llama-cpp-python (GGUF Q8_0 quantized). Produces 768-dimensional embeddings — richer representations than the initial MiniLM (384-dim), yielding better semantic matching. At ~140MB, it's efficient for deployment. No API key, no rate limits, no cost. The model uses task-specific prefixes (`search_document:` for catalogue, `search_query:` for queries) which improves retrieval accuracy.

### Why hybrid search (vector + keyword)

Pure vector search often misses exact skill name matches. For example, a query mentioning "Python" might not rank "Python (New)" in the top 10 because the embedding dilutes the keyword signal. The keyword search scans the full catalogue for exact matches and merges them into the candidate pool, catching what embeddings miss.

### Why cosine similarity via dot product

All embeddings are L2-normalized at creation time. This makes cosine similarity equivalent to a simple dot product — a single `np.dot()` call computes similarity against the entire catalogue in under 1ms. Vector search complexity is O(N·d) where N=389 and d=768, which is negligible at current catalogue size.

### Why no FAISS or vector database

The catalogue contains 389 assessments. A NumPy dot product against 389 vectors takes ~0.1ms. FAISS, ChromaDB, or Pinecone would add dependency complexity with zero performance benefit at this scale. For 100K+ assessments, a vector index would become necessary.

### Why LLM only for query parsing, not for selection

Using an LLM to directly select assessments risks hallucination — the model might recommend assessments that don't exist in the catalogue. By restricting the LLM to query understanding (extracting skills, duration, test type preferences), and using vector search for retrieval, every result is guaranteed to be a real, scraped assessment.

### Why a rule-based fallback exists

The Gemini API can fail (rate limits, network issues, key expiration). The rule-based parser uses regex and keyword matching to extract the same structured intent. It's less nuanced but deterministic. The system returns results regardless of LLM availability — it degrades gracefully, never crashes.

### Why deterministic re-ranking over LLM-based reranking

The re-ranker applies explicit, auditable rules: duration filtering, test type boosting, and balanced interleaving. This is reproducible and debuggable. An LLM-based reranker would introduce latency, cost, and non-determinism without a clear evaluation advantage at this catalogue size.

---

## API Usage

### Health Check

```bash
GET /health
```

Response:
```json
{"status": "healthy"}
```

### Get Recommendations

```bash
POST /recommend
Content-Type: application/json

{"query": "Python developer assessment under 30 minutes"}
```

Response:
```json
{
  "recommended_assessments": [
    {
      "url": "https://www.shl.com/products/product-catalog/view/python-new/",
      "name": "Python (New)",
      "adaptive_support": "No",
      "description": "Multi-choice test that measures the knowledge of Python...",
      "duration": 11,
      "remote_support": "Yes",
      "test_type": ["Knowledge & Skills"]
    }
  ]
}
```

URL inputs are also supported — the system fetches and extracts text from the page:

```bash
POST /recommend
{"query": "https://example.com/job-posting"}
```

---

## Evaluation

### Methodology

Evaluation uses **Recall@10**: what fraction of the known relevant assessments appear in the system's top 10 recommendations. Computed on the provided labelled training set.

### Metrics

| Metric | Baseline | Final |
|--------|----------|-------|
| Mean Recall@10 | 0.19 | **0.31** |
| MAP@10 | 0.11 | **0.19** |

**63% improvement** in Recall@10 through: switching to nomic-embed-text-v1.5, adding hybrid keyword search, smarter reranking with generic term filtering, and enhanced query parsing (C-suite detection, duration handling).

> **Note on recall ceiling**: 11 of 65 ground-truth URLs (17%) reference pre-packaged job solutions not present in the Individual Test Solutions catalogue. This places a hard upper bound on recall at approximately 0.83.

Average end-to-end latency (excluding cold start): ~150–300ms per request.

Run evaluation:
```bash
python -m eval.evaluate --verbose
```

Generate test predictions:
```bash
python -m eval.generate_predictions
```

### Example Queries

| Query | Top Result | Relevant? |
|-------|-----------|-----------|
| "Java developer assessment" | Java 8 (New) — Knowledge & Skills, 18 min | ✅ |
| "Leadership personality for managers" | OPQ32r — Personality & Behavior, 25 min | ✅ |
| "Python programming under 30 min" | Python (New) — Knowledge & Skills, 11 min | ✅ |

---

## Failure Modes & Improvements

### Known Limitations

- **Ambiguous queries**: Queries without clear skill keywords (e.g., "good test for hiring") produce less targeted results since the embedding has little semantic signal to match against
- **Duration data gaps**: 92 of 389 assessments have no listed duration. These assessments are never filtered out by duration constraints, which may surface irrelevant results
- **URL mismatch**: The SHL catalogue uses different URL patterns (`/solutions/products/` vs `/products/`). URL normalization handles this, but edge cases may exist in evaluation data
- **Cold start latency**: First request loads the Sentence-BERT model (~3-5 seconds). Subsequent requests are sub-second. The model is preloaded at application startup to prevent repeated loading overhead. The Dockerfile additionally pre-loads the model at build time.

### Potential Improvements

- **Cross-encoder reranking**: A cross-encoder model (e.g., `ms-marco-MiniLM`) as a second-stage reranker for more precise relevance scoring
- **Query expansion**: Use the LLM to generate 2-3 paraphrased queries and merge result sets for better recall on ambiguous queries
- **Learned boost weights**: Use training data to tune type boost, keyword boost, and balance ratios via grid search
- **Catalogue freshness**: SHL may update their catalogue. A scheduled re-scrape pipeline would keep embeddings current

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Embeddings | nomic-embed-text-v1.5 (GGUF Q8_0, 768-dim) via llama-cpp-python |
| Query parsing | Google Gemini 2.0 Flash + rule-based fallback |
| Search | Hybrid: vector search (NumPy dot product) + keyword catalogue scan |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Scraping | Selenium + BeautifulSoup |
| Deployment | Docker, Google Cloud Run / Render |

---

## Project Structure

```
├── scraper/
│   ├── scrape_catalogue.py       # Selenium-based SHL catalogue scraper
│   └── catalogue.json            # 389 scraped assessments
├── engine/
│   ├── embeddings.py             # nomic-embed-text-v1.5 encoding + persistence
│   ├── search.py                 # Hybrid vector + keyword search
│   ├── query_parser.py           # Gemini + rule-based query parsing
│   ├── reranker.py               # Filter, boost, balance test types
│   └── recommender.py            # Full pipeline orchestration
├── api/
│   └── main.py                   # FastAPI endpoints
├── frontend/
│   └── app.py                    # Streamlit UI
├── eval/
│   ├── evaluate.py               # Recall@10, MAP@10 computation
│   └── generate_predictions.py   # CSV generation for test set
├── tests/                        # 90 tests across all modules
├── data/                         # Train/test JSON (from assignment)
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Build embeddings (first time only)
python -m engine.embeddings

# Start API
python -m uvicorn api.main:app --port 8000

# Start frontend (separate terminal)
streamlit run frontend/app.py

# Run tests
python -m pytest tests/ -v

# Run evaluation (requires data/train.json)
python -m eval.evaluate --verbose
```

Set `GEMINI_API_KEY` environment variable to enable LLM-powered query parsing:
```bash
export GEMINI_API_KEY=your-key-here
```

Without it, the system uses the rule-based fallback — fully functional, just less nuanced.
