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

**Model**: `all-MiniLM-L6-v2` from Sentence-Transformers — a 22M parameter model producing 384-dimensional embeddings, optimized for semantic similarity. Runs on CPU with no external API dependency.

**Embedding text construction**: For each assessment:

```
"{name}. Test type: {', '.join(test_type)}. {description}"
```

Including `test_type` in the embedding text is deliberate. Without it, the model would only match on topic similarity (e.g., "Java" → Java tests). With it, the vector space also separates knowledge tests from personality assessments from simulations, enabling type-aware retrieval.

**Normalization**: All embeddings are L2-normalized at encode time. This converts cosine similarity to a simple dot product operation, enabling a single `np.dot(embeddings, query_vec)` call to compute all 389 similarities in ~0.1ms.

---

## 4. Query Understanding

**Primary**: Google Gemini 1.5 Flash parses the query into structured JSON:
- `job_role`: identified role
- `skills_technical`: technical skill keywords
- `skills_behavioral`: soft/behavioral skills
- `max_duration`: time constraint in minutes
- `test_types_needed`: relevant assessment categories
- `requires_balance`: `true` if both technical and behavioral skills detected

**Fallback**: If Gemini is unavailable (no API key, rate limit, network failure), a rule-based parser using regex patterns and keyword dictionaries extracts the same fields deterministically. The system never fails due to LLM unavailability.

**Balance detection** is the key insight: when a query mentions both technical skills (e.g., "Java") and behavioral skills (e.g., "collaboration"), the system activates balanced interleaving to ensure the output contains both Knowledge & Skills and Personality & Behavior assessments.

---

## 5. Re-ranking Logic

After vector search returns the top 20 candidates, the re-ranker applies three stages:

1. **Hard filter**: Remove candidates exceeding `max_duration`. Assessments with unknown duration are kept.

2. **Type boost**: Add +0.15 to similarity score for each matching test type between the candidate and `test_types_needed`. This raises relevant-type assessments above topically-similar but wrong-type results.

3. **Balance interleaving**: When `requires_balance` is active, candidates are split into technical (Knowledge & Skills, Simulations) and behavioral (Personality & Behavior, Competencies, Biodata & Situational Judgement) groups. The output interleaves roughly 50/50 from each group by score, preventing vector search's natural bias toward the dominant topic.

---

## 6. Evaluation

**Metric**: Recall@10 — fraction of known relevant assessments appearing in the system's top 10 recommendations.

**Method**: For each training query, run the full pipeline and compare returned URLs against labelled relevant URLs. URL normalization handles path variations between scraper output and ground truth.

**Additional metric**: MAP@10 (Mean Average Precision at 10) to measure ranking quality, not just presence.

Run: `python -m eval.evaluate --verbose`

---

## 7. Future Improvements

- **Cross-encoder reranking**: A cross-encoder model (e.g., `ms-marco-MiniLM`) as a second retrieval stage would improve precision at the cost of ~200ms latency per query.

- **Query expansion**: Generate 2-3 paraphrased queries via LLM and merge result sets, improving recall for ambiguous or short queries.

- **Learned weights**: Use the training data to optimize the type boost coefficient (currently 0.15) and balance ratio through grid search or Bayesian optimization.

- **Catalogue refresh pipeline**: Scheduled re-scraping to capture new assessments SHL adds to their catalogue, with delta-embedding updates.

- **FAISS index**: If the catalogue grows beyond ~10K assessments, replace NumPy dot product with a FAISS IVF index for sub-linear search time.
