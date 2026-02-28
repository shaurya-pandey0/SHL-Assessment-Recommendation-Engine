# Low-Level Design — SHL Assessment Recommendation Engine

---

## 1. Module Breakdown

```
project/
├── scraper/
│   ├── __init__.py
│   ├── scrape_catalogue.py        # Data collection
│   └── catalogue.json             # Output: 389 assessments
├── engine/
│   ├── __init__.py
│   ├── embeddings.py              # Embedding creation + persistence
│   ├── search.py                  # Vector similarity search
│   ├── query_parser.py            # LLM + rule-based query understanding
│   ├── reranker.py                # Filtering, boosting, balancing
│   └── recommender.py             # Full pipeline orchestration
├── api/
│   ├── __init__.py
│   └── main.py                    # FastAPI application
├── frontend/
│   ├── __init__.py
│   └── app.py                     # Streamlit frontend
├── eval/
│   ├── __init__.py
│   ├── evaluate.py                # Recall@10, MAP@10 computation
│   └── generate_predictions.py    # CSV output for test set
└── tests/
    ├── test_scraper.py
    ├── test_embeddings.py
    ├── test_search.py
    ├── test_query_parser.py
    ├── test_reranker.py
    ├── test_recommender.py
    └── test_api.py
```

---

## 2. Data Schema

### 2.1 Assessment (catalogue.json entry)

```json
{
  "name": "Python (New)",
  "url": "https://www.shl.com/products/product-catalog/view/python-new/",
  "remote_support": "Yes",
  "adaptive_support": "No",
  "test_type": ["Knowledge & Skills"],
  "description": "Multi-choice test that measures the knowledge of Python...",
  "duration": 11
}
```

| Field | Type | Nullable | Notes |
|-------|------|----------|-------|
| `name` | `string` | No | Assessment display name |
| `url` | `string` | No | SHL catalogue URL (unique identifier) |
| `remote_support` | `"Yes" \| "No"` | No | Remote testing support |
| `adaptive_support` | `"Yes" \| "No"` | No | Adaptive/IRT support |
| `test_type` | `string[]` | No (can be empty) | Array of canonical type names |
| `description` | `string` | No (can be `""`) | Assessment description text |
| `duration` | `int \| null` | Yes | Minutes, `null` if not listed |

### 2.2 Test Type Codes → Canonical Names

| Code | Canonical Name |
|------|---------------|
| `A` | Ability & Aptitude |
| `B` | Biodata & Situational Judgement |
| `C` | Competencies |
| `D` | Development & 360 |
| `E` | Assessment Exercises |
| `K` | Knowledge & Skills |
| `P` | Personality & Behavior |
| `S` | Simulations |

### 2.3 Parsed Query Structure

```json
{
  "job_role": "software developer",
  "skills_technical": ["python", "java"],
  "skills_behavioral": ["leadership", "collaboration"],
  "max_duration": 40,
  "test_types_needed": ["Knowledge & Skills", "Personality & Behavior"],
  "requires_balance": true
}
```

### 2.4 API Response

```json
{
  "recommended_assessments": [
    {
      "url": "string",
      "name": "string",
      "adaptive_support": "Yes|No",
      "description": "string",
      "duration": "int|null",
      "remote_support": "Yes|No",
      "test_type": ["string"]
    }
  ]
}
```

---

## 3. Module Specifications

### 3.1 `scraper/scrape_catalogue.py`

**Purpose**: Scrape all Individual Test Solutions from SHL's product catalogue.

**Key Functions**:

| Function | Signature | Description |
|----------|-----------|-------------|
| `scrape_listing_pages` | `() → list[dict]` | Selenium navigates paginated listing pages (`type=1`, 12 per page). Extracts name, URL, remote/adaptive support, test type codes from table rows. |
| `scrape_detail_page` | `(url: str) → dict` | `requests.get` on detail page. Extracts description from `og:description` meta tag, duration from page text via regex. |
| `save_catalogue` | `(assessments, path) → None` | Write JSON with `ensure_ascii=False`, indent=2. |
| `load_catalogue` | `(path) → list[dict]` | Load and return parsed JSON. |
| `validate_catalogue` | `(assessments) → dict` | Return completeness report: counts, missing fields, type distribution. |

**Pagination Logic**:
```
start=0, start=12, start=24, ... until page returns 0 new assessments.
URL: https://www.shl.com/solutions/products/product-catalog/?start={n}&type=1
```

**Duration Parsing**:
```python
# Regex: "Approximate Completion Time in minutes = 30"
pattern = r"Completion Time.*?(\d+)"
```

---

### 3.2 `engine/embeddings.py`

**Purpose**: Create, save, load, and query 384-dimensional embeddings.

**Key Functions**:

| Function | Signature | Description |
|----------|-----------|-------------|
| `get_model` | `() → SentenceTransformer` | Lazy singleton. Loads `all-MiniLM-L6-v2` once. |
| `build_embedding_text` | `(assessment: dict) → str` | `"{name}. Test type: {types}. {description}"` |
| `create_embeddings` | `(assessments, batch_size=32) → np.ndarray` | Encode all assessments. Returns `(N, 384)` float32, L2-normalized. |
| `embed_query` | `(query: str) → np.ndarray` | Encode single query. Returns `(384,)` float32, L2-normalized. |
| `save_embeddings` | `(embeddings, path) → Path` | `np.save()` to `.npy` file. |
| `load_embeddings` | `(path) → tuple[list, np.ndarray]` | Returns `(catalogue_list, embeddings_matrix)`. |
| `build_and_save` | `() → tuple` | Full pipeline: load catalogue → encode → save. |

**Embedding Text Format**:
```
"Python (New). Test type: Knowledge & Skills. Multi-choice test that measures..."
```

Including `test_type` in the text is deliberate — it encodes assessment category into the vector space, enabling type-aware retrieval without explicit filtering.

**Normalization**: `normalize_embeddings=True` at encode time. This makes cosine similarity = dot product, avoiding a per-query normalization step.

---

### 3.3 `engine/search.py`

**Purpose**: Find top-K semantically similar assessments.

**State**: Module-level globals `_catalogue` and `_embeddings`, loaded once via `_ensure_loaded()`.

| Function | Signature | Description |
|----------|-----------|-------------|
| `_ensure_loaded` | `() → None` | Load from disk if module globals are `None`. |
| `vector_search` | `(query: str, top_k=20) → list[dict]` | Embed query → dot product → argsort → return top-K with `score` field added. |
| `reload` | `() → None` | Force re-load from disk. |

**Search Algorithm**:
```python
query_vec = embed_query(query)                    # (384,)
similarities = np.dot(_embeddings, query_vec)     # (389,)
top_indices = np.argsort(similarities)[::-1][:k]  # top-K indices
```

Complexity: O(N × d) = O(389 × 384) ≈ 150K multiply-adds. Negligible.

---

### 3.4 `engine/query_parser.py`

**Purpose**: Extract structured intent from natural language.

**Dual-mode Architecture**:

```
parse_query(query)
    │
    ├── Try: _parse_with_gemini(query)
    │        └── Returns dict on success, None on failure
    │
    └── Fallback: _parse_rule_based(query)
                  └── Always returns dict (deterministic)
```

**Gemini Prompt**: Structured prompt requesting JSON output with explicit field definitions and rules. Temperature set to 0.1 for deterministic output.

**Rule-Based Fallback**:

| Extraction | Method |
|------------|--------|
| Duration | 4 regex patterns: `under N min`, `N min max`, `max N minutes`, `N-minute` |
| Technical skills | Keyword set (60+ terms): `python`, `java`, `sql`, `aws`, `docker`... |
| Behavioral skills | Keyword set (30+ terms): `leadership`, `collaboration`, `communication`... |
| Test types | Keyword → canonical name map (30+ mappings) |
| Job role | Role keyword detection with preceding qualifier |
| Balance | `True` if both `skills_technical` and `skills_behavioral` are non-empty |

**Stem Matching**: For words ≥5 chars, compare prefix of `min(len(word), len(skill)) - 2` characters. Matches `collaborates` → `collaboration`, `manages` → `management`.

---

### 3.5 `engine/reranker.py`

**Purpose**: Apply structured constraints after vector search.

| Function | Signature | Description |
|----------|-----------|-------------|
| `rerank` | `(candidates, parsed_query, top_k=10) → list[dict]` | Full re-ranking pipeline. |
| `balance_test_types` | `(candidates, top_k) → list[dict]` | Interleave technical and behavioral results. |

**Three-Stage Pipeline**:

```
Stage 1: HARD FILTER
  Remove candidates where duration > max_duration
  Keep candidates where duration is null (unknown)

Stage 2: TYPE BOOST
  For each candidate:
    overlap = candidate.test_type ∩ parsed.test_types_needed
    candidate.score += 0.15 × |overlap|

Stage 3: BALANCE (if requires_balance = true)
  Split into technical group and behavioral group
  Take ~50% from each, sorted by score within group
  Fill remaining slots from unused candidates by score
```

**Type Classification**:
- Technical: `{"Knowledge & Skills", "Simulations"}`
- Behavioral: `{"Personality & Behavior", "Competencies", "Biodata & Situational Judgement"}`

---

### 3.6 `engine/recommender.py`

**Purpose**: Orchestrate the full pipeline. Single entry point.

```python
def recommend(query: str, top_k: int = 10) -> list[dict]:
    parsed = parse_query(query)          # Step 1: understand
    candidates = vector_search(query, top_k=20)  # Step 2: retrieve
    results = rerank(candidates, parsed, top_k)   # Step 3: refine
    return format_output(results)                  # Step 4: format
```

**Output Formatting**: Strips internal fields (e.g., `score`), returns only the 7 required SHL fields.

---

### 3.7 `api/main.py`

**Endpoints**:

| Method | Path | Request | Response | Status |
|--------|------|---------|----------|--------|
| `GET` | `/health` | — | `{"status": "healthy"}` | 200 |
| `POST` | `/recommend` | `{"query": "..."}` | `{"recommended_assessments": [...]}` | 200 |
| `POST` | `/recommend` | `{"query": ""}` | `{"detail": "Query cannot be empty"}` | 400 |

**URL Detection**: If `query.startswith("http")`, fetch page via `requests`, extract text with BeautifulSoup, use extracted text as the query.

**Lifespan**: Embeddings are pre-loaded at startup via `asynccontextmanager` to avoid cold-start latency on first request.

---

### 3.8 `eval/evaluate.py`

**Metrics**:

```python
Recall@K = |recommended ∩ relevant| / |relevant|

AP@K = (1/min(|relevant|, K)) × Σ (precision_at_i × is_relevant_i)

Mean Recall@K = (1/Q) × Σ recall_i

MAP@K = (1/Q) × Σ AP_i
```

**URL Normalization**: Strips trailing slashes, lowercases, handles path variations between scraper output and ground truth data.

---

## 4. Concurrency & State Management

- **Module-level singletons**: `_catalogue`, `_embeddings` (in `search.py`), `_model` (in `embeddings.py`), `_gemini_client` (in `query_parser.py`) are loaded once and shared across all requests.
- **Thread safety**: NumPy dot product and model inference are read-only operations on immutable data. No locks needed.
- **Stateless API**: No session state, no database writes. Each request is independent.

---

## 5. Error Handling Strategy

| Layer | Error | Handling |
|-------|-------|----------|
| Query Parser | Gemini API failure | Catch all exceptions → return `None` → fallback activates |
| Query Parser | Invalid JSON from Gemini | `json.loads` try/except → return `None` → fallback |
| Query Parser | Missing fields in Gemini response | Field validation → return `None` → fallback |
| API | Empty query | `HTTPException(400)` |
| API | URL fetch failure | `HTTPException(400)` with detail message |
| API | Recommendation pipeline crash | `HTTPException(500)` with generic message (no stack trace leak) |
| Embeddings | Model download failure | Logged at startup; subsequent requests will fail with clear error |
| Embeddings | `.npy` file missing | `FileNotFoundError` with instruction to run `python -m engine.embeddings` |

---

## 6. Testing Strategy

| Test File | Scope | Count | Dependencies |
|-----------|-------|-------|-------------|
| `test_scraper.py` | Type mapping, validation, save/load, data structure | 19 | None (unit) |
| `test_embeddings.py` | Text construction, shape, normalization, similarity | 12 | Sentence-BERT |
| `test_search.py` | Result count, sorting, semantic relevance, fields | 9 | Sentence-BERT |
| `test_query_parser.py` | Duration, types, skills, balance, integration | 22 | None (unit) |
| `test_reranker.py` | Duration filter, type boost, balance, combined | 9 | None (unit) |
| `test_recommender.py` | Output fields, types, relevance, edge cases | 10 | Sentence-BERT |
| `test_api.py` | Health, recommend, error handling, response format | 9 | FastAPI TestClient |
| **Total** | | **90** | |

Tests that require Sentence-BERT use `scope="module"` fixtures to load the model once per test file, keeping the full suite under 30 seconds.

---

## 7. Configuration

| Parameter | Location | Default | Override |
|-----------|----------|---------|----------|
| Embedding model | `engine/embeddings.py` | `all-MiniLM-L6-v2` | Change `MODEL_NAME` constant |
| Embedding dimension | Determined by model | 384 | N/A |
| Vector search top-K | `engine/search.py` | 20 | `vector_search(query, top_k=N)` |
| Final result count | `engine/recommender.py` | 10 | `recommend(query, top_k=N)` |
| Type boost coefficient | `engine/reranker.py` | 0.15 | Change in `rerank()` |
| Gemini model | `engine/query_parser.py` | `gemini-2.0-flash` | Change in `_parse_with_gemini()` |
| Gemini temperature | `engine/query_parser.py` | 0.1 | Change in `_parse_with_gemini()` |
| API port | CLI argument | 8000 | `uvicorn api.main:app --port N` |
| API key | `.env` file | None | Set `API_KEY=...` in `.env` |

---

## 8. File Sizes & Memory Footprint

| Asset | Size | Loaded at |
|-------|------|-----------|
| `catalogue.json` | ~180 KB | Startup |
| `embeddings.npy` | ~600 KB | Startup |
| Sentence-BERT model | ~80 MB | First query (or startup if pre-loaded) |
| Total RAM at runtime | ~200 MB | — |
