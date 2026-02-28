# High-Level Design — SHL Assessment Recommendation Engine

---

## 1. System Overview

The SHL Assessment Recommendation Engine is a retrieval-based system that maps natural language hiring queries to relevant SHL product catalogue assessments. It is not a generative system — every recommendation is a real, scraped assessment from SHL's catalogue.

```
┌──────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                            │
│  ┌──────────────┐             ┌──────────────────────────┐       │
│  │  Streamlit UI │             │  Direct API (curl/httpx) │       │
│  └──────┬───────┘             └────────────┬─────────────┘       │
│         │                                  │                     │
│         └──────────────┬───────────────────┘                     │
│                        ▼                                         │
│              ┌─────────────────┐                                 │
│              │  FastAPI Server  │  POST /recommend, GET /health  │
│              └────────┬────────┘                                 │
│                       ▼                                          │
│         ┌─────────────────────────┐                              │
│         │   Recommendation Engine  │                             │
│         │  ┌───────────────────┐  │                              │
│         │  │  Query Parser     │──┼── Gemini 2.0 Flash (or       │
│         │  │  (LLM + fallback) │  │   rule-based fallback)       │
│         │  └────────┬──────────┘  │                              │
│         │           ▼             │                              │
│         │  ┌───────────────────┐  │                              │
│         │  │  Vector Search    │──┼── Sentence-BERT embeddings   │
│         │  │  (cosine sim)     │  │   (389 × 384 matrix)         │
│         │  └────────┬──────────┘  │                              │
│         │           ▼             │                              │
│         │  ┌───────────────────┐  │                              │
│         │  │  Re-ranker        │  │                              │
│         │  │  + Type Balancer  │  │                              │
│         │  └────────┬──────────┘  │                              │
│         └───────────┼─────────────┘                              │
│                     ▼                                            │
│            Top 10 JSON Results                                   │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                     OFFLINE PIPELINE                             │
│                                                                  │
│  ┌───────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │ Selenium       │───▶│ catalogue.json│───▶│ embeddings.npy   │  │
│  │ Scraper        │    │ (389 items)   │    │ (389 × 384)      │  │
│  └───────────────┘    └──────────────┘    └──────────────────┘   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. Major Components

| Component | Responsibility | Technology |
|-----------|---------------|------------|
| **Data Scraper** | Crawl SHL catalogue, extract assessment metadata | Selenium + BeautifulSoup |
| **Embedding Engine** | Encode assessments into dense vectors | Sentence-BERT (`all-MiniLM-L6-v2`) |
| **Query Parser** | Extract structured intent from user query | Google Gemini 2.0 Flash + regex fallback |
| **Vector Search** | Find semantically similar assessments | NumPy dot product (cosine similarity) |
| **Re-ranker** | Apply hard filters, boost scores, balance types | Rule-based scoring |
| **API Layer** | Serve recommendations via REST | FastAPI + Uvicorn |
| **Frontend** | Web UI for querying | Streamlit |
| **Evaluation** | Compute Recall@10, MAP@10 on labelled data | Custom Python module |

---

## 3. Data Flow

### 3.1 Offline (One-time)

```
SHL Website ──[Selenium]──▶ Listing Pages ──[parse rows]──▶ Assessment URLs
                                                                  │
Assessment URLs ──[requests]──▶ Detail Pages ──[extract]──▶ catalogue.json
                                                                  │
catalogue.json ──[Sentence-BERT]──▶ embeddings.npy (389 × 384)
```

### 3.2 Online (Per request)

```
User Query
    │
    ├──[is URL?]──▶ fetch HTML ──▶ extract text ──▶ query_text
    │
    ▼
Query Parser ──▶ {job_role, skills, max_duration, test_types, requires_balance}
    │
    ▼
Sentence-BERT ──▶ query_vector (384-dim)
    │
    ▼
np.dot(embeddings, query_vector) ──▶ similarity scores (389 values)
    │
    ▼
Top 20 candidates ──▶ Re-ranker ──▶ Top 10 results ──▶ JSON response
```

---

## 4. Deployment Architecture

```
┌──────────────────────┐       ┌──────────────────────┐
│  Streamlit Cloud     │       │  Render / Cloud Run   │
│  (Frontend)          │──────▶│  (FastAPI Backend)     │
│                      │ HTTP  │                        │
│  Port: 8501          │       │  Port: 8080            │
└──────────────────────┘       │  Docker Container      │
                               │  ├── catalogue.json    │
                               │  ├── embeddings.npy    │
                               │  └── SBERT model cache │
                               └──────────────────────┘
```

- **No database required.** All data lives in two files: `catalogue.json` (180KB) and `embeddings.npy` (600KB).
- **No external vector DB.** NumPy handles all similarity computations in-process.
- **Gemini API** is the only external dependency at runtime, and it's optional (fallback exists).

---

## 5. Scalability Considerations

| Dimension | Current | If scaled |
|-----------|---------|-----------|
| Catalogue size | 389 assessments | FAISS IVF index at 10K+ |
| Search latency | ~0.1ms (NumPy dot) | ~1ms (FAISS) |
| Embedding model | CPU inference (~50ms) | GPU / batched inference |
| Query parsing | Gemini API (500ms) | Cached common queries |
| Concurrent users | Uvicorn async workers | Kubernetes horizontal scaling |
| Data freshness | Manual re-scrape | Scheduled cron pipeline |

---

## 6. Failure Modes & Mitigation

| Failure | Impact | Mitigation |
|---------|--------|------------|
| Gemini API down / rate-limited | Query parsing degrades | Rule-based fallback activates automatically |
| Gemini returns invalid JSON | Parsing fails | JSON validation + fallback |
| SHL website structure changes | Scraper breaks | Scraper is run offline; cached data continues to serve |
| Embedding model download fails | System won't start | Model pre-downloaded in Docker build |
| Out-of-vocabulary query | Low similarity scores | Sentence-BERT handles unseen text via subword tokenization |

---

## 7. Security

- **API key storage**: Gemini API key in `.env` file, excluded from Git via `.gitignore`
- **No user data stored**: System is stateless; no queries are persisted
- **CORS**: Open (`*`) for development; restrict to frontend origin in production
- **Input validation**: Pydantic models enforce request schema
- **URL fetching**: Timeout + error handling prevents SSRF-style abuse
