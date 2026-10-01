Great questions. Let me answer each one by pointing to your actual code.

---

### 1. "Are the 10 results the top 10?"

**Yes.** The pipeline starts wide and narrows down:
- Vector search pulls **top 50** candidates
- Role-focused search adds ~20 more
- Keyword search adds more
- You end up with a pool of **~50-80 candidates**
- The reranker filters, boosts scores, and returns the **top 10 by final score**

The `top_k=10` is passed through the whole chain. See `recommender.py` line 181:
```python
def recommend(query: str, top_k: int = 10) -> list[dict]:
```
And the reranker at line 275:
```python
results = rerank(candidates, parsed, top_k=top_k)
```

---

### 2. "Confirm — Gemini is ONLY for query parsing?"

**Correct. Gemini does ONE thing: parse the query into structured JSON.** Nothing else.

It does NOT:
- Search for assessments
- Rank results  
- Compress queries
- Pick the final recommendations

The proof is in `query_parser.py` line 269-289:
```python
def parse_query(query: str) -> dict:
    # Try Gemini first
    result = _parse_with_gemini(query)
    if result is not None:
        return result
    # Fallback to rule-based
    return _parse_rule_based(query)
```

Gemini returns a dict like:
```json
{
  "job_role": "java developer",
  "skills_technical": ["java"],
  "skills_behavioral": ["collaboration"],
  "max_duration": 40,
  "test_types_needed": ["Knowledge & Skills"],
  "requires_balance": false
}
```

That's it. That dict then gets used by the rest of the pipeline.

---

### 3. "How does Gemini parse the query? Do we give it instructions?"

**Yes — there's a detailed prompt.** Look at `query_parser.py` lines 35-60:

```python
GEMINI_PROMPT = '''Analyze this hiring/assessment query and extract structured information.
Return ONLY valid JSON, no markdown fences, no other text.

Query: "{query}"

Return this JSON structure:
{{
  "job_role": "string or null",
  "skills_technical": ["list of technical skills mentioned"],
  "skills_behavioral": ["list of soft/behavioral skills mentioned"],
  "max_duration": integer or null,
  "test_types_needed": ["list from: Knowledge & Skills, Personality & Behavior, ..."],
  "requires_balance": true or false
}}

Rules:
- If the query mentions both technical AND soft skills, set requires_balance to true
- If only technical: test_types_needed = ["Knowledge & Skills"]
- If duration limit mentioned (e.g. "under 30 minutes"), extract as integer minutes
...'''
```

So you tell Gemini: *"Here's the query, extract these exact fields, follow these rules."* Gemini returns JSON, your code validates it has all required fields (lines 111-128), and passes it downstream.

---

### 4. "Query compression — is Gemini involved?"

**No. Query compression is separate code, no LLM involved.**

It's in `recommender.py` lines 104-176, function `compress_query()`. It only kicks in for **long queries (>80 words)** — like a full job description pasted in.

What it does (purely with regex and keyword lists):
- Extracts job role via regex patterns
- Finds known technologies (python, java, aws, etc.) from a hardcoded set
- Finds behavioral terms (leadership, collaboration, etc.)
- Combines them into a short focused string

Example: a 500-word JD becomes → `"senior software engineer java python aws leadership Knowledge & Skills"`

For short queries like `"java developer test"`, compression is skipped entirely (line 119):
```python
if len(words) <= WORD_THRESHOLD:  # 80
    return query  # return unchanged
```

---

### 5. "When Gemini isn't there, how does parsing happen?"

The rule-based parser in `query_parser.py` function `_parse_rule_based()` (line 231) does the same job using:

| What it extracts | How |
|---|---|
| **Technical skills** | Checks query against a hardcoded set of ~50 skills: `python, java, sql, react, aws...` (line 141-153) |
| **Behavioral skills** | Checks against another set: `leadership, communication, teamwork, cultural fit...` (line 156-167) |
| **Duration** | Regex patterns like `under (\d+) min`, `about an hour` → 60 (line 206-213) |
| **Test types** | Keyword mapping: `"coding"` → Knowledge & Skills, `"personality"` → Personality & Behavior (line 170-204) |
| **Job role** | Looks for words like `developer, analyst, manager, coo, ceo` (line 215-225) |
| **Executive detection** | If COO/CEO/CTO found → auto-adds "leadership" + "Personality & Behavior" (line 248-253) |

**What it CAN'T do that Gemini can:**
- Understand `"express.js"` implies JavaScript/Node.js
- Infer skills not explicitly written (e.g. "full-stack" → implies both frontend and backend)
- Handle creative phrasings like "someone who can crunch numbers" → analytics

**What it handles fine:**
- `"Python developer test under 30 min"` → skills: [python], duration: 30, types: [K&S] ✅
- `"Leadership assessment for VP"` → detects VP as executive, adds personality ✅
- `"Java and teamwork, 40 minutes"` → both tech + behavioral, requires_balance: true ✅

---

### Summary — the complete honest picture:

```
Query: "Java developer who collaborates, under 40 min"
                    ↓
        ┌──── Gemini (if working) ────┐
        │  Understands nuance         │
        │  Returns structured JSON    │
        └─────────────────────────────┘
                    ↓ (or if Gemini fails)
        ┌──── Rule-based parser ──────┐
        │  Regex + keyword matching   │
        │  Returns SAME JSON format   │
        └─────────────────────────────┘
                    ↓
   Parsed: {skills_tech: ["java"], skills_behav: ["collaboration"],
            max_duration: 40, requires_balance: true}
                    ↓
   compress_query() → skipped (query is short)
                    ↓
   Vector search (50) + Role search (20) + Keyword search
                    ↓
   Reranker: filter >40min → boost K&S → boost "java" → balance tech/behavioral
                    ↓
   TOP 10 returned
```

Both paths produce the **exact same JSON structure** → the rest of the pipeline doesn't know or care which parser was used. That's the graceful degradation.