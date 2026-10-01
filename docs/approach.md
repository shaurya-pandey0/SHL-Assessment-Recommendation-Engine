# SHL Assessment Recommendation Engine — Approach Document

---

## 1. Problem Definition

Given a natural language query describing a hiring need (e.g., "Java developer who collaborates with external teams, assessment under 40 minutes"), the system recommends up to 10 relevant SHL assessments from their product catalogue. It handles diverse query types — technical skill assessments, personality evaluations, cognitive ability tests — and balances recommendations when a query involves both technical and behavioral dimensions.

The system exposes a REST API (`POST /recommend`, `GET /health`), a Streamlit web frontend, and produces evaluation metrics (Recall@10, MAP@10) demonstrating retrieval quality.

**Live deployment**: API at `http://34.59.156.213:8000`, Frontend at `http://34.59.156.213:8501`

---

## 2. Data Collection

**Source**: SHL Product Catalogue at `shl.com/solutions/products/product-catalog/`

**Method**: Two-stage scraping pipeline:

1. **Listing pages** (Selenium): Headless Chrome navigates all paginated listing pages for Individual Test Solutions (`type=1`). Each row yields: assessment name, URL, remote/adaptive support, and test type codes (A, B, C, D, E, K, P, S).

2. **Detail pages** (requests + BeautifulSoup): Each assessment's detail page is fetched to extract: description text (from `og:description` meta tag) and duration in minutes (from "Approximate Completion Time").

**Result**: 389 individual assessments with fields: `name`, `url`, `remote_support`, `adaptive_support`, `test_type` (array), `description`, `duration`.

**Validation**: 100% description coverage, 76% duration coverage. Test type distribution: Knowledge & Skills (243), Personality & Behavior (79), Simulations (47), Ability & Aptitude (42), Biodata & Situational Judgement (28), Competencies (13), Development & 360 (7), Assessment Exercises (2).

---

## 3. Embedding Strategy

**Model**: `nomic-embed-text-v1.5` (GGUF format, f32 precision) — a 137M parameter model producing 768-dimensional embeddings, run locally via llama-cpp-python. Chosen over the initial `all-MiniLM-L6-v2` (384-dim) for richer semantic representations while remaining CPU-friendly. No API key, no rate limits, no cost.

**Embedding text construction**: Each assessment is embedded as:
```
"search_document: {name}. Test type: {types}. {description}. Keywords: {domain_hints}"
```

The `search_document:` / `search_query:` prefixes are required by nomic-embed to distinguish document vs query embeddings. Including `test_type` in the text separates knowledge tests from personality assessments in vector space. Each test type appends domain-specific keyword hints (e.g., "Ability & Aptitude" → "cognitive reasoning analytical aptitude intelligence") to nudge embeddings toward the right semantic cluster.

All embeddings are L2-normalized at creation time, making cosine similarity equivalent to a dot product — O(N·d) search against 389 vectors completes in <1ms.

---

## 4. Query Understanding

**Primary**: Google Gemini 2.0 Flash parses the query into structured JSON: `job_role`, `skills_technical`, `skills_behavioral`, `max_duration`, `test_types_needed`, `requires_balance`.

**Fallback**: If Gemini is unavailable (no API key, rate limit, network failure), a deterministic rule-based parser using regex patterns and keyword dictionaries extracts the same fields. The system never fails due to LLM unavailability — it degrades gracefully.

**Key parser enhancements** (implemented during optimization):
- Executive title detection (COO, CEO, CTO, VP, Director) → auto-adds "leadership" and "Personality & Behavior"
- Natural duration parsing: "about an hour" → 60 min, "1-2 hours" → 60 min
- Cultural fit terms ("right fit", "culture", "organizational fit") → triggers personality assessment detection

---

## 5. Hybrid Retrieval Pipeline

The system uses a **three-stage hybrid retrieval** approach:

### Stage 1: Vector Search (semantic)
- Short queries (<80 words): single vector search, top 50 candidates
- Long JDs (>80 words): dual search — compressed query (key skills/role extracted) + original text, merged by max score. Compression prevents embedding dilution from JD boilerplate.
- **Role-focused search**: additional vector search using just job role + top skills as a focused query, surfacing role-specific assessments diluted in full query embeddings

### Stage 2: Keyword Search (exact matching)
- Scans full catalogue for exact skill keyword matches in assessment names and descriptions
- Uses specific technology/domain terms (e.g., "java", "sql", "marketing", "leadership")
- Generic terms ("engineering", "development", "software") filtered out to prevent noisy matches
- Results merged into candidate pool — catches what embeddings miss (e.g., "Python" → "Python (New)")

### Stage 3: Re-ranking and Balancing
1. **Hard filter**: Remove candidates exceeding `max_duration` (unknown duration assessments kept)
2. **Type boost**: +0.20 per matching test type between candidate and query's `test_types_needed`
3. **Keyword boost**: +0.20 per specific skill keyword in candidate text, +0.15 extra for name matches
4. **Smart balance**: When both technical and behavioral skills detected, guarantees minimum 2 slots for minority group (not rigid 50/50). Prevents over-allocating when query is predominantly one type.

---

## 6. Optimization Journey

### Baseline (MiniLM 384-dim + basic vector search)
| Metric | Score |
|--------|-------|
| Mean Recall@10 | 0.19 |
| MAP@10 | 0.11 |

### Error Analysis
- **Theoretical max recall is ~0.83**: 11 of 65 ground-truth URLs point to pre-packaged job solutions not in the Individual Test Solutions catalogue
- Many relevant assessments ranked 20-100+ in vector search — outside top-10 cutoff
- Exact skill matches missed: "Python" in query didn't find "Python (New)" via embeddings alone
- Parser blind spots: "COO" unrecognized, "about an hour" unparsed, "cultural fit" undetected

### Improvement 1: Upgrade to nomic-embed-text-v1.5 (768-dim)
Richer embeddings improved semantic matching. **Recall: 0.19 → 0.20.**

### Improvement 2: Hybrid retrieval + wider candidate pool
Added keyword search for exact skill name matches. Widened vector search from top-20 to top-50 candidates. Brought skill-named assessments ("JavaScript (New)", "SQL Server (New)") into the pool. **Recall: 0.20 → 0.24.**

### Improvement 3: Smarter reranking
- Filtered generic terms from keyword matching to reduce noise
- Increased type boost (0.15→0.20) and keyword boost (0.20/0.15)
- Changed balance from rigid 50/50 to guarantee-minimum-2 for minority group
- **Recall: 0.24 → 0.26**

### Improvement 4: Enhanced query parser
- C-suite title detection (COO → leadership/personality)
- "About an hour" duration parsing
- Cultural fit terms added to behavioral keyword set
- Q3 (COO cultural fit query) went from 0.0 → 0.50 recall
- **Recall: 0.26 → 0.31**

### Final Results
| Metric | Baseline | Final | Improvement |
|--------|----------|-------|-------------|
| Mean Recall@10 | 0.19 | **0.31** | +63% |
| MAP@10 | 0.11 | **0.19** | +73% |

---

## 7. Deployment

**Infrastructure**: Google Cloud Platform (4 vCPU, 15GB RAM, Debian 11)

**API**: FastAPI + Uvicorn, containerized with Docker. The Docker image includes the GGUF model, pre-computed embeddings, and all dependencies. The embedding model is loaded once at startup; subsequent requests are sub-second.

**Frontend**: Streamlit, running on the same instance, connecting to the API at localhost.

**Tech Stack**:
| Component | Technology |
|-----------|------------|
| Embeddings | nomic-embed-text-v1.5 (GGUF f32, 768-dim) via llama-cpp-python |
| Query parsing | Google Gemini 2.0 Flash + rule-based fallback |
| Search | Hybrid: vector search (NumPy dot product) + keyword catalogue scan |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Scraping | Selenium + BeautifulSoup |
| Deployment | Docker on GCP Compute Engine |

---

## 8. Future Improvements

- **Cross-encoder reranking**: A model like `ms-marco-MiniLM` as a second-stage reranker for more precise relevance scoring (~200ms added latency)
- **Query expansion**: Generate 2-3 paraphrased queries via LLM and merge result sets for better recall on ambiguous queries
- **Learned boost weights**: Use training data to optimize type/keyword boost coefficients via grid search
- **Catalogue refresh pipeline**: Scheduled re-scraping to capture new assessments with delta-embedding updates
