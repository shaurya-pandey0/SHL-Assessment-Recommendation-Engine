# SHL Assessment Recommendation Engine — Approach Document

---

## 1. Problem Definition

Given a natural language query describing a hiring need (e.g., "Java developer who collaborates with external teams, assessment under 40 minutes"), the system must recommend up to 10 relevant SHL assessments from their product catalogue. The system must handle diverse query types — technical skill assessments, personality evaluations, cognitive ability tests — and balance recommendations when a query involves both technical and behavioral dimensions.

The implemented system exposes a REST API (`POST /recommend`, `GET /health`), includes a Streamlit web frontend, and produces evaluation metrics demonstrating retrieval quality.

---

## 2. Data Collection

**Source**: SHL Product Catalogue at `shl.com/solutions/products/product-catalog/`

**Method**: Two-stage scraping pipeline:

1. **Listing pages** (Selenium): The catalogue renders via JavaScript, so headless Chrome navigates all paginated listing pages for Individual Test Solutions (`type=1`). Each listing row yields: assessment name, URL, remote testing support, adaptive/IRT support, and test type codes (letter codes: A, B, C, D, E, K, P, S).

2. **Detail pages** (requests + BeautifulSoup): Each assessment's detail page is fetched server-side to extract: full description text (from `og:description` meta tag), assessment duration in minutes (from "Approximate Completion Time"), and supplementary metadata.

**Result**: 389 individual assessments with fields: `name`, `url`, `remote_support`, `adaptive_support`, `test_type` (array of full names), `description`, `duration`.

**Validation**: 100% description coverage, 76% duration coverage (92 assessments have no listed duration on SHL's site). Test type codes mapped to canonical names: Knowledge & Skills (243), Personality & Behavior (79), Simulations (47), Ability & Aptitude (42), Biodata & Situational Judgement (28), Competencies (13), Development & 360 (7), Assessment Exercises (2).

---

## 3. Embedding Strategy

**Model**: `nomic-embed-text-v1.5` (GGUF Q8_0 quantized) — a 137M parameter model producing 768-dimensional embeddings, run locally via llama-cpp-python. This model was chosen over the initial `all-MiniLM-L6-v2` (384-dim) because it produces richer, higher-quality representations for retrieval tasks while still running efficiently on CPU.

**Embedding text construction**: For each assessment:

```
"search_document: {name}. Test type: {types}. {description}. Keywords: {domain_hints}"
```

The `search_document:` prefix is required by nomic-embed to distinguish document embeddings from query embeddings (which use `search_query:` prefix). Including `test_type` in the text is deliberate — it separates knowledge tests from personality assessments in vector space, enabling type-aware retrieval.

**Domain enrichment**: Each test type appends short keyword hints (e.g., "Ability & Aptitude" adds "cognitive reasoning analytical aptitude intelligence") to nudge embeddings toward the right semantic cluster.

**Normalization**: All embeddings are L2-normalized at encode time, converting cosine similarity to a dot product for fast retrieval.

---

## 4. Query Understanding

**Primary**: Google Gemini 2.0 Flash parses the query into structured JSON:
- `job_role`: identified role (including C-suite titles: COO, CEO, CTO)
- `skills_technical`: technical skill keywords
- `skills_behavioral`: soft/behavioral skills
- `max_duration`: time constraint in minutes
- `test_types_needed`: relevant assessment categories
- `requires_balance`: `true` if both technical and behavioral skills detected

**Fallback**: If Gemini is unavailable (no API key, rate limit, network failure), a rule-based parser using regex patterns and keyword dictionaries extracts the same fields deterministically. The system never fails due to LLM unavailability.

**Key enhancements to the parser** (implemented during optimization):
- Executive title detection (COO, CEO, CTO, VP, Director) auto-adds "leadership" as a behavioral skill and "Personality & Behavior" as a needed test type
- Duration parsing handles natural phrasing: "about an hour" → 60 min, "1-2 hours" → 60 min
- Cultural fit terms ("right fit", "culture", "organizational fit") trigger personality assessment detection

---

## 5. Hybrid Retrieval Pipeline

The system uses a **three-stage hybrid retrieval** approach — not just vector search alone:

### Stage 1: Vector Search (semantic)
- For short queries (<80 words): single vector search, top 50 candidates
- For long JDs (>80 words): dual search (compressed query + original text), merged by max score
- Additionally, a **role-focused search** using just the job role + skills as a focused query, surfacing role-specific assessments that get diluted in full JD embeddings

### Stage 2: Keyword Search (exact matching)
- Scans the full 389-assessment catalogue for exact skill keyword matches in names and descriptions
- Uses specific technology/domain terms extracted from the query (e.g., "java", "sql", "marketing", "leadership")
- Generic terms ("engineering", "development", "software") are filtered out to prevent noisy matches
- Results are merged into the candidate pool, adding assessments that vector search missed entirely

### Stage 3: Re-ranking and Balancing
1. **Hard filter**: Remove candidates exceeding `max_duration`. Assessments with unknown duration are kept.
2. **Type boost**: +0.20 to score for each matching test type between candidate and `test_types_needed`.
3. **Keyword boost**: +0.20 per specific skill keyword found in candidate text, +0.15 extra for name matches. Generic terms excluded to prevent noise.
4. **Smart balance**: When both technical and behavioral skills detected, guarantees minimum 2 slots for the minority group (not a rigid 50/50 split). This prevents over-allocating to behavioral slots when the query is predominantly technical.

---

## 6. Optimization Journey

### Initial baseline (MiniLM embeddings + basic pipeline)
| Metric | Score |
|--------|-------|
| Mean Recall@10 | 0.19 |
| MAP@10 | 0.11 |

### Key observations from error analysis
- **Theoretical max recall is 0.83**: 11 of 65 ground-truth URLs point to pre-packaged job solutions not in the 389 Individual Test Solutions catalogue
- **Many relevant assessments ranked 20-100+** in vector search — too low to make the top-10 cutoff
- **Exact skill matches missed**: "Python" in query should find "Python (New)" assessment, but embedding similarity alone diluted this
- **Parser blind spots**: "COO" not recognized as a job role, "about an hour" not parsed as a duration, "cultural fit" not detected as a behavioral indicator

### Improvement 1: Switch to nomic-embed-text-v1.5 (768-dim)
Richer embeddings improved semantic matching. Recall went from 0.19 → 0.20.

### Improvement 2: Hybrid retrieval (keyword search + wider candidate pool)
Added keyword-based catalogue search for exact skill name matches. Widened vector search from top-20 to top-50 candidates. This brought assessments like "JavaScript (New)", "SQL Server (New)" into the candidate pool.

### Improvement 3: Smarter reranking
- Filtered out generic terms ("engineering", "software") from keyword matching to reduce noise
- Increased type boost to 0.20 and keyword boost to 0.20/0.15
- Changed balance from rigid 50/50 to guarantee-minimum-2 for minority group
Recall improved to 0.26.

### Improvement 4: Enhanced query parser
- Added C-suite title detection (COO → leadership/personality)
- Fixed "about an hour" duration parsing
- Added cultural fit terms to behavioral keyword set
This brought Q3 (COO cultural fit) from 0.0 → 0.50 recall.

### Final results
| Metric | Score |
|--------|-------|
| Mean Recall@10 | 0.31 |
| MAP@10 | 0.19 |

**63% improvement** in Recall@10 over the baseline.

---

## 7. Future Improvements

- **Cross-encoder reranking**: A cross-encoder model (e.g., `ms-marco-MiniLM`) as a second retrieval stage would improve precision at the cost of ~200ms latency per query.

- **Query expansion**: Generate 2-3 paraphrased queries via LLM and merge result sets, improving recall for ambiguous or short queries.

- **Learned weights**: Use the training data to optimize the type boost coefficient and balance ratio through grid search or Bayesian optimization.

- **Catalogue refresh pipeline**: Scheduled re-scraping to capture new assessments SHL adds to their catalogue, with delta-embedding updates.
