# SHL Assessment Recommendation Engine: Interview Answers

<p class="meta">Based on the repository's docs and source code. Section 10 is skipped because it requires running the code. Where an answer describes your working process, reword it to match what you actually did.</p>

## 1. Project overview

### 1. Give a 2-minute introduction: problem, input/output, user

**Problem.** SHL's catalogue has hundreds of assessments (I scraped 389 Individual Test Solutions). A recruiter hiring a "Java developer who works with business teams, 40 minutes max" has to work out which mix of skill, personality and aptitude tests fits and stays under the time limit. Keyword filters miss intent: "collaborates with business teams" implies a behavioural test even though no test is named.

**User.** Recruiters, hiring managers and HR/talent teams choosing assessments for a role.

**Input.** A plain-English hiring need, a full job description, or a job-description URL, sent to `POST /recommend` or entered in the Streamlit UI.

**Output.** Up to 10 real SHL assessments as JSON, each with url, name, description, duration, remote support, adaptive support and test type.

**How it works.** An LLM (Gemini) turns the query into structured intent, with a rule-based fallback. Three retrieval paths build a candidate pool: vector search with a local nomic-embed model, a role-focused vector search, and keyword matching. A rule-based re-ranker then applies the duration limit, test-type and keyword boosts, and the technical/behavioural balance.

**Results.** On the 10 labelled queries, Recall@10 went from 0.19 to 0.31 and MAP@10 from 0.11 to 0.19, over four rounds of error analysis. The ceiling is about 0.83, because 11 of the 65 correct answers aren't in the catalogue. The system is served with FastAPI and Streamlit, in Docker, on a GCP VM.

### 1.1 Walk through the architecture, from /recommend to JSON

1. **Input check.** Strip the query. An empty query returns 400. If it starts with `http://` or `https://`, fetch the page and use its text as the query.
2. **Parse.** `parse_query` tries Gemini first and falls back to the rule-based parser. Output: `job_role, skills_technical, skills_behavioral, max_duration, test_types_needed, requires_balance`.
3. **Compress.** If the query has more than 80 words, `compress_query` builds a 10-20 word query from role, known technologies, behavioural terms and test types.
4. **Retrieve.** Three paths:
    - Main vector search, top 50. For long queries it runs on both the compressed and the original text, and results are merged by maximum score.
    - Role-focused vector search, top 20.
    - Keyword search, top 30.

    Only new URLs are added to the pool.

5. **Re-rank.** Hard duration filter, then +0.20 per matching test type, then +0.20 per skill keyword hit plus 0.15 per name hit. If `requires_balance` is set, the test-type balance is applied.
6. **Format.** Return the 7 required fields and drop internal fields such as `score`. Pydantic validates the response.

### 1.1.1 How is URL input handled differently from plain text?

- Detection: the whole query must start with `http://` or `https://`. A URL inside a sentence is not fetched.
- `extract_text_from_url` runs `requests.get` with a 10 s timeout, then `raise_for_status`. BeautifulSoup (lxml) removes `script, style, nav, header, footer`, extracts the text and truncates it to 5000 characters.
- After that it follows the same pipeline as text. In practice the extracted text is almost always over 80 words, so it goes through compression and the dual search.
- Differences from plain text: an extra network call of up to 10 s, a 5000-character cap that plain text doesn't have, and the risk of noisy page text such as menus and cookie banners.

### 1.1.1.1 How does the system behave if the URL fetch fails?

- Network error, timeout or non-2xx status: the exception is caught and the API returns **HTTP 400** with "Could not fetch URL content: &lt;error&gt;".
- The page loads but yields no text: **HTTP 400**, "Could not extract text from URL".
- There is no retry and no fallback, such as treating the URL as plain text. Streamlit shows "Bad request: ...".
- **Silent failure:** JavaScript-rendered pages, login walls and cookie walls return 200 with little or irrelevant text. The pipeline runs on that text and returns weak results without any error.
- Improvements:
    - Detect low-content pages and ask the user to paste the text instead.
    - Render pages in a headless browser.
    - Don't echo raw exception text to the client.
    - Add SSRF protection: block private, loopback and metadata IPs after DNS resolution, and re-check on redirects.

### 1.1.2 What does the output schema look like, and why are the field names fixed?

```
{"recommended_assessments": [
  {"url": "...", "name": "Python (New)", "adaptive_support": "No",
   "description": "...", "duration": 11, "remote_support": "Yes",
   "test_type": ["Knowledge & Skills"]}
]}
```

- Enforced by Pydantic `response_model`. `duration` is `Optional[int]` (null when unknown) and `test_type` is always a list.
- The names are fixed because the assignment specified this exact response format (the code says "exact fields required by SHL"). Evaluators and clients parse it without any mapping. Internal fields such as `score` are removed.
- Changing a name would be a breaking API change, so it would need a versioned endpoint.

### 1.1.3 Which parts are deterministic, and which are not?

**Deterministic:**

- the catalogue and embeddings (fixed files)
- embedding a given text with the same model file
- vector search, keyword search, re-ranking and balance
- the rule-based parser
- output formatting

**Not deterministic:**

- **Gemini output.** Temperature 0.1 is low, not zero, and the model can change on Google's side.
- **Gemini availability.** Success versus fallback produces a different parse, so the same query can return different results.
- **URL content.** Pages change over time.
- **A small edge case.** The stem matcher loops over a Python `set` of strings and stops at the first match. Set order can differ between processes, so a word that matches two skills could pick a different one on different runs.

Since Gemini 2.0 Flash was shut down (June 2026), the system runs only on the fallback, so in practice it is now fully deterministic.

### 1.2 How many assessments are in the catalogue, and how did you check the scrape was complete?

**389 assessments, all with unique URLs.** Checks:

- Pagination runs `start=0,12,24,...` with `type=1` and stops when a page has no table rows or no new URLs (at most 35 pages).
- URLs are de-duplicated.
- `validate_catalogue` reports:
    - the total against the target of at least 377 (PASS)
    - description coverage: 100%
    - duration coverage: 76% (297 of 389; 92 missing)
    - type distribution: Knowledge & Skills 243, Personality & Behavior 79, Simulations 47, Ability & Aptitude 42, Biodata & SJT 28, Competencies 20, Development & 360 7, Assessment Exercises 2 (51 items have more than one type)

Gap: I didn't check the count against a total shown on the site. I'd add that check and diff the results of repeat scrapes.

### 1.2.1 Why only Individual Test Solutions, not pre-packaged job solutions?

- The task was to recommend individual assessments, and the scraper targets `type=1` (Individual Test Solutions) only.
- Pre-packaged solutions are bundles of individual tests for a job family, such as "Professional 7.1 Solution" or "Entry Level Sales 7.1".
- Trade-off: 11 of the 65 ground-truth URLs are pre-packaged solutions, which caps recall at about 0.83. Adding them as a separate type is on my improvements list.

### 1.2.2 Why a two-stage scraper (Selenium for listings, requests for detail pages)?

- **Listing pages are rendered by JavaScript.** A plain GET returns no table rows, so the scraper uses headless Chrome. It waits for `table tbody tr`, then uses JavaScript to extract the name, link, remote and adaptive indicators, and test-type codes.
- **Detail pages are rendered on the server.** `requests` with BeautifulSoup is much faster and lighter than a browser. It extracts:
    - the description from `og:description`, with the "Name: " prefix removed
    - the duration from "Approximate Completion Time in minutes = N", falling back to "N minutes" when the value is between 1 and 300
- The browser is used only where it's needed (about 33 listing pages), not for all 389 detail pages. Delays of 1 s per listing page and 0.3 s per detail page keep the load on the site polite.

## 2. Intent extraction (LLM parsing)

### 2. What is intent extraction, which fields, and which model?

Intent extraction turns free text into structured constraints that retrieval and re-ranking can act on.

- **Fields:** `job_role`, `skills_technical`, `skills_behavioral`, `max_duration` (int or null), `test_types_needed` (from the 8 canonical SHL types), `requires_balance` (bool).
- **Model:** `gemini-2.0-flash`, called through the `google-genai` SDK with temperature 0.1 and max_output_tokens 500.
- **API key:** read from `GEMINI_API_KEY`, `GOOGLE_API_KEY` or `API_KEY`, with `.env` supported.
- **Fallback:** the rule-based parser, which returns the same schema.
- **Current status:** Google shut down Gemini 2.0 Flash on June 1, 2026, so the fallback is now handling every request.

### 2.1 Why that model?

- The task is short, structured extraction, not deep reasoning. A fast, low-cost "Flash" model fits that.
- It runs on every request, so low latency matters more than raw capability.
- It follows a JSON-only output format reliably.
- The LLM is swappable: any model that returns the same JSON works, because the rest of the pipeline only sees the parsed dict.

### 2.1.1 What's the latency and cost impact of calling Gemini on every request?

- **Latency.** It's one sequential call before retrieval, so it sits on the critical path. The docs estimate about 500 ms, compared with about 0.1 ms for search and tens of ms per embedding. Gemini dominates request time. The README's 150-300 ms figure fits requests that don't wait on Gemini.
- **Cost.** Each call sends a ~400-token prompt plus the query (up to ~800 words for long job descriptions) and returns at most 500 output tokens. That's small per request, but it grows linearly with traffic.
- **When Gemini is down.** Every request still tries the call and waits for the error, which adds latency for no benefit. There's no circuit breaker.
- **Improvements:**
    - cache parses of repeated queries
    - add a circuit breaker and a timeout
    - read the model name from config

### 2.2 Walk through the prompt. Why JSON only, and why temperature 0.1?

1. The task: "Analyze this hiring/assessment query and extract structured information."
2. "Return ONLY valid JSON, no markdown fences, no other text."
3. The query is inserted.
4. A JSON template with types. `test_types_needed` is restricted to the 8 canonical names, which are exactly the strings used in the catalogue, so the type boost's set intersection works.
5. Rules:
    - technical and soft skills both present: `requires_balance` = true and include both types
    - only technical: `["Knowledge & Skills"]`
    - only behavioural: `["Personality & Behavior", "Competencies"]`
    - cognitive terms: add Ability & Aptitude
    - a duration limit becomes integer minutes; no limit means null
    - `job_role` should be the specific role

**Why JSON only:** the response goes straight into `json.loads`, with no free-text parsing, and the fields match the rule-based output one to one. Downstream code doesn't care which parser ran.

**Why 0.1:** extraction should be repeatable. The same query should give the same constraints, which keeps evaluation reproducible and reduces invented skills. Zero would work equally well.

**Risk:** the query is inserted into the prompt as-is, so prompt injection is possible. The damage is limited because the output is validated and the LLM never chooses the assessments. Gemini's structured-output mode (a response schema) would be stronger than relying on the prompt alone.

### 2.2.1 How do you validate the response? What if a field is missing or wrongly typed?

**Validation steps:**

1. strip code fences
2. `json.loads`
3. check that all 6 required keys are present
4. coerce list fields to lists
5. cast `max_duration` to int

**Outcomes:**

- **Missing field:** returns `None` and the whole parse falls back to the rule-based parser. The two results are not merged.
- **A list field that isn't a list:** silently replaced with `[]`.
- **`max_duration` that can't be cast to int:** set to `None`.

**Gaps:**

- The type of `job_role` isn't checked.
- `requires_balance` isn't coerced to a bool, so the string "false" would count as true.
- Test-type names aren't checked against the 8 canonical names. An invented name simply never matches.
- There's no range check on duration.

**Fix:** a Pydantic model for the parsed intent, or Gemini's response-schema mode.

### 2.3 Why use the LLM only for parsing, not for picking assessments?

- An LLM picking tests can hallucinate names or URLs, and it doesn't know the current catalogue.
- Putting all 389 items in the prompt would be slow and costly, and the results would vary from run to run.
- Splitting the work guarantees every result is a real, scraped assessment and keeps the ranking auditable. The LLM does what it's best at, understanding messy language, and deterministic retrieval does the selecting.

### 2.4 When exactly does the system fall back, and how does it behave when Gemini is down?

**The fallback runs when:**

- no API key is set (no network call is made)
- the SDK import or client creation fails
- the call raises any exception: network, authentication, quota or rate limit, model not found or retired
- the response isn't valid JSON
- required fields are missing

**When Gemini is down:** the client object is cached, so every request tries the call, fails, logs a warning and uses the rule-based parse. The schema is the same, so nothing downstream changes, and the API never fails because of the LLM.

**Today:** Gemini 2.0 Flash is retired, so every request with a key takes the exception path and pays the latency of a failed call. The fix is to move the model name to config and add a circuit breaker.

### 2.4.1 If Gemini and the rule-based parser disagree, which is more likely right?

**Gemini is usually right on meaning:**

- implied intent ("handle difficult clients" suggests a behavioural test)
- plurals and paraphrases ("Java developers")
- full role titles: "Senior Data Analyst", where the rule-based parser returns just "senior"
- ignoring ordinary words such as "it" or "data", which the rule-based parser counts as skills

**The rule-based parser is more reliable on literal constraints.** It never invents a skill or duration, and its behaviour is predictable.

**How I'd combine them:** use Gemini for role, skills and types, and cross-check duration with the regex. I didn't measure how often they agree. The way to do it: log both parses for the training queries and compare them.

### 2.5 How is requires_balance decided, and what does it change?

- **Rule-based:** true only if both `skills_technical` and `skills_behavioral` are non-empty.
- **Gemini:** set by the prompt rule (both technical and soft skills mentioned).
- **Downstream:** it only changes the final selection. The re-ranker calls `balance_test_types` instead of taking the top 10 by score. Retrieval and boosts are unchanged.
- **Caveat:** false-positive skills ("it", "data", "development" matched from "developer", "management") make balance trigger more often than it should.

## 3. Rule-based parser details

### 3. How does the rule-based parser extract the job role? One example.

1. Lowercase the query and split it into words with `[\w-]+`.
2. Scan left to right for the first word in `JOB_ROLES`. That list has about 35 entries: developer, engineer, analyst, manager, director, and so on, plus the seniority words junior, senior, graduate, intern, and the C-suite titles coo, ceo, cto and similar.
3. If there's a previous word and it isn't a, an, the, for or as, return "previous word + role word". Otherwise return the role word alone.

**Example:** "looking for a python developer": the first role word is "developer" and the previous word is "python", so the result is **"python developer"**.

### 3.1 For "I want to hire a Senior Data Analyst...", what does _extract_job_role return, and why?

It returns **"senior"**:

- "senior" is itself in `JOB_ROLES`, and it comes before "analyst", so it's the first hit.
- The previous word, "a", is a stopword, so the function returns just "senior".

**Impact:** the real role is lost. The role-focused query starts with "senior" instead of "senior data analyst". (This query is only 27 words, so compression isn't involved.)

**Fixes:**

- skip seniority words and keep scanning
- or use the recommender's `ROLE_PATTERNS` regex, which matches "senior data analyst"
- or take the longest match

Gemini would have returned "Senior Data Analyst".

### 3.1.1 What about multi-word roles like "QA Engineer" or "Assistant Admin"?

Only one word before the role word is kept:

| Query text | Result | Note |
|---|---|---|
| "QA Engineer" | "qa engineer" | Correct |
| "ICICI Bank Assistant Admin" | "bank assistant" | "admin" isn't a role word, so it's dropped |
| "Full stack developer" | "stack developer" | "full" is dropped |
| "Senior software engineer" | "senior" | Same seniority problem as 3.1 |
| "Java developers" (train query 1) | none | Plurals aren't matched |

**Fixes:**

- use the `ROLE_PATTERNS` regex, which handles multi-word titles
- lemmatise plurals
- prefer the longest match
- add missing words such as "admin"

### 3.2 How does executive-title detection (COO, CEO, VP) change the parsed output?

If any word is in {coo, ceo, cto, cfo, cmo, cio, vp, director, executive}:

- "leadership" is added to `skills_behavioral`
- "Personality & Behavior" is added to `test_types_needed`

**Effects:**

- Personality tests get the type boost, and "leadership" gets the keyword boost.
- If technical skills are also present, `requires_balance` becomes true, because it's computed after this step.
- The role comes out as "coo", since "coo" is also in `JOB_ROLES`.

**Subtlety:** Competencies is only added if behavioural skills were found *before* this check. If the title is the only behavioural signal, Competencies is left out.

**False positive:** "executive" is in the set, so entry-level titles such as "Sales Executive" or "Customer Service Executive" also get "leadership".

### 3.3 How does duration extraction work? Walk through "about an hour" and "1-2 hours".

Checks run in this order:

1. The regex for "about / around / approximately (an) hour" returns **60**.
2. `(\d+)(?:\s*-\s*\d+)?\s*hours?` returns the **first** number × 60.
3. The minute patterns, in order:
    - "under / less than / max / within / up to / no more than N min"
    - "N min or less / max"
    - "duration / time / length N"
    - "N-min" or "N min"

**Examples:**

- **"about an hour" gives 60.** It's a target, but it's treated as a maximum.
- **"1-2 hours" gives 60.** It takes the lower bound, which is stricter than the user meant (120). Train query 10 ("1-2 hour long") also gives 60.
- **Practical impact: none.** The longest duration in the catalogue is 60 minutes, so a limit of 60 or 120 filters exactly the same items.

### 3.3.1 For "30-40 mins" (ICICI Assistant Admin), which number is picked? Is it right? Effect?

The query is: "...test should be 30-40 mins long".

- Neither hour regex matches, and nor do the first three minute patterns.
- The pattern `(\d+)[-\s]?min...` is searched left to right. At "30" the next characters are "-40", not "min", so it fails there. The first match is "40 mins", which gives **40**.
- **40 is correct for a maximum:** the user accepts tests of up to 40 minutes.
- But it's right **by accident** of how the regex scans, not because ranges are handled on purpose. For example, "between 30 min and 40 min" would give 30.
- **Effect:** tests over 40 minutes are removed; tests of 40 minutes or less and tests with unknown duration are kept. If it had picked 30, valid 31-40 minute tests would have been dropped.
- "0-2 years" isn't misread, because the hour regex requires the word "hour".
- **Fix:** an explicit range regex that takes the upper bound.

### 3.4 Explain the stem-like fuzzy matching in _extract_skills. Where can it give false positives?

For a word that isn't an exact skill, with at least 5 characters:

1. Compare it with each single-word skill of at least 5 characters.
2. `prefix_len = min(len(word), len(skill)) - 2`, and it must be at least 5. In effect, both words need 7 or more characters.
3. If the first `prefix_len` characters are equal, add the skill and stop.

**Works as intended:**

- "collaborates" matches "collaboration" ("collaborat")
- "managers" matches "management" ("manage")
- "culturally" matches "culture" ("cultu")

**False positives:**

| Word | Matched skill | Why it's wrong |
|---|---|---|
| "community" | "communication" | Different meaning |
| "personal" (as in "personal data") | "personality" | Different meaning |
| "developer" | "development" | Becomes a technical skill |
| "technician" | "technical" | Becomes a technical skill |
| "engineer" | "engineering" | Becomes a technical skill |

**Exact-match false positives** from very short list entries: "it", "go", "r", "ai" and "data" are also ordinary English words ("make it happen", "on the go").

**Consequences:** wrong test types, balance triggered unnecessarily, and noisy boosts.

**Fixes:**

- a real stemmer or lemmatiser (Porter, spaCy)
- guards for short skills, such as matching "IT" only in uppercase
- context rules

## 4. Embeddings

### 4. Which embedding model? Exact name, dimension, format.

- **Model:** `nomic-embed-text-v1.5`, 768-dimensional.
- **Format:** GGUF. The default file is `nomic-embed-text-v1.5.Q8_0.gguf` (~140 MB). The code accepts any `nomic-embed-text-v1.5*.gguf`.
- **Runtime:** runs on CPU through `llama-cpp-python` (`embedding=True`, `n_ctx=2048`).
- **Catalogue matrix:** 389 × 768 float32, stored in `scraper/embeddings.npy`.

### 4.1 Why nomic-embed-text-v1.5 over MiniLM, and what changed in the metrics?

- **Baseline:** `all-MiniLM-L6-v2` (384-dimensional Sentence-BERT).
- **Why nomic:**
    - it's trained for retrieval, with separate query and document prefixes
    - it has a much longer context (MiniLM truncates short inputs; nomic here uses a 2048-token window)
    - its 768-dimensional vectors are richer
    - it still runs on CPU as GGUF, with no API
- **Metrics:** recall went from 0.19 to only **0.20 (+0.01)**. The lesson: the bottleneck was retrieval design and parsing, not the encoder.
- **Costs:** a larger model file, a C++ build for llama-cpp in Docker, and slower embedding.

### 4.2 What are the search_document: / search_query: prefixes, and why do they matter?

- Nomic was trained with task prefixes: documents are embedded as `"search_document: ..."` and queries as `"search_query: ..."`.
- This teaches the model that a query and a document play different roles (a short question versus a longer description).
- Leaving the prefixes out, or mixing them up, lowers retrieval quality. They must also be the same at build time and query time.
- In the code, `build_embedding_text` adds the document prefix and `embed_query` adds the query prefix.

### 4.3 What text goes into each assessment's embedding? Why add test type and keyword hints?

The format is: `search_document: {name}. Test type: {types}. {description}. Keywords: {hints}`

**Hints per test type**, for example:

| Test type | Hint words |
|---|---|
| Ability & Aptitude | cognitive reasoning analytical aptitude intelligence |
| Personality & Behavior | personality behavior collaboration communication culture |
| Knowledge & Skills | technical skills expertise proficiency tool |

**Why:** names and descriptions are short and product-like ("Verify G+"). The type words give queries such as "cognitive test" or "culture fit" something to match, and they encode the category into the vectors without a hard filter.

**Risk:** items of the same type become more similar to each other, which makes it harder to tell them apart. The hints were chosen by hand and weren't tested separately.

### 4.3.1 Why truncate document text to 1500 characters?

- The code comment says it's to stay within the 2048-token context. 1500 characters is only about 300-400 tokens, so it's a conservative safety margin. Descriptions are short, so it rarely matters.
- **Related, for queries:** llama-cpp-python's `embed()` truncates input to `n_batch` tokens by default (`n_batch` defaults to 512). So when the original text of a very long job description is embedded, only roughly its first 512 tokens are used. That's another reason compression helps.

### 4.4 Why L2-normalize the vectors, and how does that relate to cosine similarity?

- cosine(a, b) = a·b / (|a||b|). For unit vectors the denominator is 1, so **cosine equals the dot product**.
- The catalogue is normalised once at build time and each query once per request. Search is then a single `np.dot` of (389 × 768) with (768,).
- Scores all fall in [-1, 1] on one consistent scale. `_normalize` guards against zero-length vectors.

### 4.5 Are catalogue and query embeddings guaranteed to come from the same model? How would you check?

**Not guaranteed:**

- `embeddings.npy` is built offline and saved with no metadata.
- At runtime, the model file is chosen by glob: the first sorted `nomic-embed-text-v1.5*.gguf`.

**What can go wrong:**

- **A different model altogether** (for example an old 384-dimensional MiniLM file): the shapes don't match and `np.dot` raises an error. This failure is loud.
- **The same model at a different quantization** (built with f32, served with Q8_0): same 768 dimensions, so there's small, silent drift.
- **Catalogue and matrix out of sync:** row counts and order aren't checked, so misaligned rows would silently return the wrong assessments.

**How to check:**

- Save a metadata file next to the embeddings: model filename and hash, dimension, catalogue hash, build date. Assert it at load time.
- Assert `shape == (len(catalogue), 768)`.
- At startup, re-embed a few catalogue items and check the cosine with the stored rows is about 1.0 (above 0.99).

The Dockerfile rebuilds the embeddings at image build time when the model file is present, which keeps them in sync inside the container.

### 4.6 Why no FAISS or vector database?

- Brute force over 389 × 768 is about 300K multiply-adds: well under a millisecond, and exact (no approximation).
- A vector database adds a service, a dependency, index builds and operations work, for no gain at this size.
- I'd switch at roughly 10K-100K+ items, or when metadata filtering and frequent updates matter. Options: FAISS (HNSW or IVF) in-process, or pgvector or Qdrant.

## 5. Hybrid retrieval

### 5. Explain hybrid search. What does each source contribute?

| Source | Size | Contributes |
|---|---|---|
| Main vector search | top 50 | Semantic matches: paraphrases and concepts. Long queries search both the compressed and the original text. |
| Role-focused vector search | top 20 | Role-specific tests that get diluted in a full-query embedding |
| Keyword search | top 30 | Exact skill names the embeddings miss ("Python (New)", "SQL Server (New)") |

The pool is the union by URL, up to about 100 candidates, and all of them go to re-ranking. Adding hybrid search and widening the pool from 20 to 50 raised recall from 0.20 to 0.24.

### 5.1 What is the role-focused search, and why add it?

- **Query:** `job_role + test_types_needed + top 5 technical skills + top 3 behavioural skills + " assessment"`. It uses vector search, top 20, and adds only new URLs.
- **Why:** a full-query embedding mixes many signals (company, benefits, both kinds of skills), so tests for a specific role such as sales, admin or customer service rank low. A short, focused query asks directly which assessments fit this role and these skills.
- It runs for any query that has parsed fields, not only long ones.
- **Limitation:** it depends on the parser. For example, the rule-based role "senior" for train query 10 weakens it.

### 5.2 For long JDs you run dual search. Why merge by max, not average?

- Both lists are cosine scores from the same model, so they're directly comparable. That makes max meaningful here.
- Max works like an OR: an item that strongly matches either view stays high.
- An average would penalise items that one view finds strongly and the other weakly. That's exactly the case where compression rescues something diluted in the full text.
- Many items appear in only one of the two top-50 lists, so there's no second score to average.
- **Downside:** agreement between the two views isn't rewarded. Alternatives: Reciprocal Rank Fusion, or max plus a small bonus when both views find the item.

### 5.3 How is the keyword search score calculated?

`score = (hits + name_hits) / (2 × number_of_skills)`

- `hits` = the number of skills that appear as substrings in name + description
- `name_hits` = the number of skills that appear in the name
- The range is 0-1. Results are sorted and the top 30 kept.

**Example:** skills [python, sql]. "Python (New)" contains "python" in both its name and description, so hits = 1 and name_hits = 1, giving (1 + 1) / 4 = **0.5**.

### 5.3.1 Vector scores are cosines and keyword scores are hit ratios. How can the two be compared?

- **Strictly, they can't.** They use different scales and mean different things.
- A single-skill query that matches a test's name gives a keyword score of 1.0. Cosine scores for relevant items are usually well below 1.
- **Why it still works in practice:**
    - keyword results are only added when the URL isn't already in the pool, so most candidates keep a vector score
    - re-ranking applies the same boosts to everyone
- **Problems:**
    - keyword-only items can be over-ranked
    - the keyword signal is counted twice: once in the base score and again in the re-ranker's keyword boost
- **Better options:**
    - Reciprocal Rank Fusion (sum of 1/(k + rank))
    - normalise each source before merging
    - simplest: give keyword-found items their cosine score, which is already computed for all 389 items

### 5.3.2 When a URL shows up from both sources, which score is kept?

The first one seen. The order is:

1. main vector search (for long queries, the max of the two vector scores)
2. role-focused search, which only adds new URLs
3. keyword search, which only adds new URLs

So a vector-found item keeps its cosine score and the keyword score is discarded. The keyword signal still reaches it through the re-ranker's keyword boost.

### 5.3.3 Why filter out generic terms like "software" and "data"?

- Words such as software, data, development, testing, management, security and cloud appear in a large share of names and descriptions.
- Matching them pulls in or boosts much of the catalogue equally. That doesn't discriminate between tests; it just adds noise that pushes out specific matches.
- Filtering them out keeps the boosts for specific terms (java, sql, excel, marketing). This was part of the 0.24 to 0.26 step.
- The parser still uses these words for detecting test types and balance.
- **Code smell:** the same list is defined twice (`GENERIC_TERMS` in the recommender, `GENERIC_SKILLS` in the re-ranker). It should be one shared constant.

### 5.3.4 Keyword matching uses substrings. Why is it noisy, and how would you measure that?

**Why it's noisy:** `s in text` is a plain substring test.

| Skill | Also matches |
|---|---|
| "java" | "javascript" |
| "excel" | "excellent" |
| "go" | "good", "category" |
| "r" | almost every description |

"R developer" is the worst case: keyword search returns 30 arbitrary items, and nearly every candidate gets a +0.20 boost.

**How to measure it:**

1. For each skill term, measure the share of catalogue items it matches. A specific term should match only a few; flag any that match more than about 5%.
2. Measure the precision of the keyword path on the training data: of the candidates added only by keyword search, how many are relevant?
3. Run ablations: recall and MAP with keyword search and the boost on versus off, and with word-boundary matching versus substring matching.

**Fixes:** word-boundary regex (`\b`), plus an allowlist or minimum length for short skills (r, go, c).

## 6. Scoring and re-ranking

### 6. How are search scores and rule-based scores combined?

1. **Hard duration filter first.**
2. Compute the final score:

   `final = base + 0.20 × (matching test types) + 0.20 × (specific skills found in name + description) + 0.15 × (skills found in the name)`

   `base` is the cosine score from vector search, or the keyword ratio for items found only by keyword search.
3. Take the top 10 by score, or apply `balance_test_types` if balance is required.

The rule-based scores are added on top of the search similarity; they nudge it rather than replace it.

### 6.1 Walk through the final score. Where do 0.20 and 0.15 come from?

**Example** (illustrative): a Java knowledge test for a Java query.

| Component | Value |
|---|---|
| cosine score | 0.55 |
| type match (Knowledge & Skills) | +0.20 |
| "java" in name and description | +0.20 |
| "java" in the name | +0.15 |
| **Total** | **1.10** |

**Where the numbers come from:**

- They were **hand-tuned** on the 10 training queries.
- The type boost started at 0.15 in v1 and was raised to 0.20. The keyword boosts were set to 0.20 and 0.15.
- Cosine scores of the top candidates are close together, so 0.20 is deliberately large: a correct type or an exact skill should usually beat a slightly higher semantic score.
- A skill in the name gets extra weight because that's the strongest signal ("Python (New)").
- They weren't optimised systematically. The fix is a grid search with leave-one-out cross-validation.

### 6.1.1 Can a strong keyword match push out a semantically better result? Example?

**Yes. Examples:**

- **Java vs JavaScript.** For "Java developer...", the skill "java" also matches "JavaScript (New)" as a substring. It gets +0.35, the same boost as the real Java tests, and can push out a relevant behavioural or aptitude test with a higher cosine score.
- **"excel" vs "excellent".** "excel" matches "excellent" in unrelated descriptions.
- **Many listed skills.** An item whose description lists several of the query's technologies collects +0.20 for each one and can outrank a closer single-skill match.

**Mitigations:**

- word boundaries
- a cap on the total keyword boost
- weight by term rarity (IDF)
- only boost items that already have a minimum semantic score

### 6.2 How does the duration hard filter work? Why keep unknown durations?

- **How it works:** if `max_duration` is set, keep a candidate only if its duration is null or at most the limit. It's applied before the boosts, and failing items are removed completely.
- **Why keep unknowns:** 92 of 389 items (24%) have no duration on SHL's site. Dropping them would lose many valid tests, including relevant ones, and hurt recall.
- **Trade-off:**
    - some long tests get through
    - two items (English Comprehension (New), Spelling (U.S.) (New)) are listed with 0 minutes, so they pass every limit
- **Alternatives:**
    - a soft penalty for unknown duration
    - estimate durations from similar tests
    - treat 0 as unknown

### 6.3 Explain balance_test_types. Why a minimum of 2 rather than 50/50?

**How it works:**

1. Split candidates into three groups:
    - **technical:** has Knowledge & Skills or Simulations
    - **behavioural:** otherwise has Personality & Behavior, Competencies or Biodata & SJT
    - **other:** everything else, such as Ability & Aptitude only
2. Sort each group by score. If one of the first two groups is empty, return the rest by score.
3. Minority slots = `min(size, max(2, top_k // 4))`, which is **2** for a top 10. The majority group gets the remaining **8**. Any leftover slots are filled from the remaining candidates by score.

**Why not 50/50:** v1 used 50/50. But most queries are mainly technical: of the relevant items that are in the catalogue, the training labels have 42 technical vs 8 behavioural. Forcing 5 behavioural slots replaced strong technical matches with weak behavioural ones. A minimum of 2 guarantees coverage without over-allocating. This was part of the 0.24 to 0.26 step.

**Weaknesses:**

- An item with both technical and behavioural types counts as technical.
- When the majority group can fill its 8 slots, there are no leftover slots, so "other" items (for example, the 20 pure Ability & Aptitude tests) never appear. A needed cognitive test can be crowded out.
- The ratio is fixed, not based on the query.

**Fixes:**

- a three-way split driven by `test_types_needed`
- a ratio based on the query's mix of skills

## 7. Evaluation metrics

### 7. Define Recall@10 and walk through one query.

**Recall@10 = |top 10 ∩ relevant| / |relevant|** per query, averaged over all queries. Order is ignored, and URLs are normalised before comparing.

**Train query 1** (Java developers who collaborate, 40 minutes) has 5 relevant items: Automata Fix, Core Java (Entry Level), Java 8, Core Java (Advanced Level) and Interpersonal Communications.

*Illustrative:* if the top 10 contains Java 8, Core Java (Entry Level) and Interpersonal Communications, recall = 3/5 = **0.60**. The actual result requires running the code.

### 7.1 Define MAP@10. Why report it next to recall?

- **AP@10** = (1 / min(|relevant|, 10)) × the sum of precision@i over each rank i ≤ 10 that holds a relevant item. precision@i = hits in the top i / i.
- **MAP@10** is the mean of AP@10 over all queries.

**Example:** 5 relevant items, with hits at ranks 1, 3 and 8.

- P@1 = 1, P@3 = 2/3, P@8 = 3/8
- AP = (1 + 0.667 + 0.375) / 5 = **0.41**
- recall = 0.60

**Why report both:**

- Recall measures coverage (did we find them?). MAP measures ranking (are they near the top?).
- Recruiters read the top few results, so ranking matters. Two systems with the same recall can be very different to use.
- MAP rising from 0.11 to 0.19 shows re-ranking moved relevant items up, not just into the list.

### 7.2 How many queries are in the train set? How reliable is a mean over that few?

- **Data:** 10 labelled queries with 65 relevant URLs (5-10 each), plus 9 unlabelled test queries.
- **The mix:**
    - 6 short queries (8-38 words)
    - 4 long job descriptions (257-498 words)
- **Reliability is low.** One query moving by 0.5 shifts the mean by 0.05. That's bigger than some of my individual improvements (+0.01, +0.02).
- **How to present it:** treat the gains as a direction, not proof. Report per-query results, bootstrap confidence intervals over the queries, and collect more labels.

### 7.2.1 Did you tune on the data you evaluated on? What's the overfitting risk?

**Yes.** Weights, thresholds, keyword lists and parser rules were all tuned on the same 10 queries I report, with no held-out split.

**Risk:** rules fitted to specific queries. The COO, cultural-fit and "about an hour" fixes all target train query 3. So 0.31 is optimistic for unseen queries.

**What I did about it:**

- kept the rules general: all C-suite titles, general duration phrases
- used the test set only for generating predictions

**The proper approach:**

- leave-one-out cross-validation: tune on 9, evaluate on 1, rotate
- a frozen held-out set
- testing every new rule on unseen queries

### 7.3 How were test-set predictions generated, and what format did the CSV need?

- `python -m eval.generate_predictions` loads `data/test.json` (9 queries, 17-784 words) and runs `recommend(query, top_k=10)` for each one.
- It writes `predictions.csv` with the columns **Query, Assessment_url**: one row per recommendation, with the query text repeated. That gives **90 rows**, the format the assignment required.
- **Risk worth mentioning:** the CSV uses catalogue URLs (`/products/product-catalog/view/...`), but the labels use `/solutions/products/product-catalog/view/...`. My evaluation normalises them. If the grader compared exact strings, correct answers would be missed. I'd output URLs in the label format, or confirm that the grader normalises them.

## 8. Ground truth

### 8. How was ground truth established, and who created the labels?

- The labels were provided with the assignment by SHL (`data/train.json`). I didn't create them, and I don't know SHL's exact labelling process.
- They reflect SHL's product knowledge, such as standard bundles and combinations for a role, not just semantic similarity. That's one reason pure vector search underperforms.

### 8.1 What does one train entry look like?

```
{"query": "I am hiring for Java developers who can also collaborate
           effectively with my business teams. Looking for an
           assessment(s) that can be completed in 40 minutes.",
 "relevant_urls": [
   ".../solutions/products/product-catalog/view/automata-fix-new/",
   ".../view/core-java-entry-level-new/", ".../view/java-8-new/",
   ".../view/core-java-advanced-level-new/",
   ".../view/interpersonal-communications/"]}
```

Every entry has the keys `query` and `relevant_urls`, with 5-10 URLs each.

### 8.2 Why do some ground-truth URLs not exist in your catalogue?

- 11 of the 65 point to **pre-packaged job solutions**, which are bundles rather than Individual Test Solutions. Examples:
    - `professional-7-1-solution` (appears in 3 queries)
    - `entry-level-sales-7-1`
    - `sales-representative-solution`
    - `manager-8-0-jfa-4310`
- The scraper collects `type=1` (individual tests) only, so these can never be recommended.

### 8.2.1 The recall ceiling is about 0.83. How did you work that out?

**Calculation:**

1. Normalise every ground-truth URL and look it up in `catalogue.json`.
2. 54 of 65 exist, and 54/65 = **0.83**. That's the pooled figure.

**Per query:**

- Q2 (sales graduates) is capped at 0.56, and Q7 (ICICI admin) at 0.67.
- Averaged per query, which is how Mean Recall works, the ceiling is about **0.85**.

**Either way,** 0.31 is roughly 37% of what's achievable.

### 8.3 How did you handle /solutions/products/ vs /products/, and why normalise?

**`normalize_url`:**

1. strip whitespace and the trailing "/"
2. lowercase
3. replace `solutions/products/product-catalog/` with `products/product-catalog/`

It's applied to both the recommended URLs and the relevant URLs.

**Why normalise:**

- The catalogue URLs and the label URLs point to the same assessments, written differently.
- Comparing exact strings would count every correct answer as a miss, giving recall of about 0.
- Normalising compares which assessment it is, not how the URL is formatted.

**Not handled:** slug changes, query strings, http vs https. Comparing slugs would be more robust.

### 8.4 If you had to build ground truth yourself, how would you do it?

1. **Collect** realistic queries: real job descriptions across roles, levels and lengths.
2. **Label:** SHL consultants or I/O psychologists mark relevant assessments, ideally graded (must-have / useful / not relevant).
3. **Pool the candidates** (TREC-style): take the top-k from several systems plus expert picks, and have experts judge the combined pool. This reduces bias toward any one system.
4. **Check quality:** at least 2 annotators per query, measure agreement (Cohen's kappa) and resolve disagreements.
5. **Split and refresh:** split train / validation / test by query, and refresh when the catalogue changes.
6. **In production:** use recruiter behaviour (which recommendations get chosen) as implicit labels, correcting for position bias.

## 9. Error analysis

### 9. How did you do error analysis to keep improving recall?

1. Run `python -m eval.evaluate --verbose`. It prints each query's Recall@10, AP@10, hit count and first 3 missed URLs.
2. Start with the worst queries and work out why each miss happened.
3. Change one component.
4. Re-run the full evaluation and keep the change if mean recall improves.

That was four rounds: embeddings, then hybrid retrieval, then re-ranking, then the parser.

### 9.1 Given the missed URLs, how did you tell why each was missed?

Check each miss in this order:

1. **Not in the catalogue?** Look up the normalised URL in `catalogue.json`. If it's missing, it's pre-packaged and can't be fixed. That's how the 11 were found.
2. **Not retrieved?** Check its rank in raw vector search over all 389 items, and whether the keyword or role-focused path found it. If none did, it's a retrieval problem.
3. **Filtered out?** Compare its duration with the parsed `max_duration`. If it was removed, it's a parser or filter problem.
4. **Ranked too low?** If it survived the filter but sits below 10th, compare its final score with the 10th item's and see which boost it lacked, such as a type that wasn't requested or a skill that wasn't parsed. That points to a ranking or parser problem.

`evaluate.py` prints only the misses. The pool and rank checks come from inspecting each stage; the recommender logs the parsed query and candidate counts.

### 9.1.1 Where did relevant items rank in raw vector search?

- Many sat at about **rank 20-100+**. They were retrieved, but below the top 10, so the original top-20 pool couldn't include them at all.
- Exact-name matches were also missed. For example, "Python" didn't surface "Python (New)".
- **What that led to:**
    - widening the pool to 50
    - adding the keyword and role-focused paths
    - using boosts to lift relevant items into the top 10

### 9.2 What were the baseline and final numbers, and what caused each improvement?

| Stage | Change | Recall@10 |
|---|---|---|
| Baseline | MiniLM 384-d, plain vector search, top 20 | 0.19 |
| 1 | nomic-embed-text-v1.5, 768-d | 0.20 |
| 2 | Keyword search, and pool widened from 20 to 50 | 0.24 |
| 3 | Generic-term filter, boosts to 0.20/0.15, balance changed to "min 2" | 0.26 |
| 4 | Parser: C-suite titles, "about an hour", cultural-fit terms | 0.31 |

MAP@10 went from 0.11 to 0.19 overall. MAP wasn't recorded for each stage.

### 9.2.1 Which change gave the biggest gain, and which was surprisingly small?

- **Biggest:** the parser fixes (+0.05). Hybrid retrieval was a close second (+0.04).
- **Surprisingly small:** the embedding model upgrade (+0.01), despite doubling the dimension. The problem was what was being searched and how the results were combined, not the encoder.

### 9.2.2 How did you make sure a fix for one query didn't hurt the others?

**What I did:**

- re-ran the full evaluation on all 10 queries after every change
- compared the per-query table, not just the mean
- kept fixes general: classes of rules, not specific query strings
- relied on the 90 unit tests to catch behaviour regressions in the parser and re-ranker

**Honestly,** with 10 queries that's weak protection. A held-out set and cross-validation are the proper fix.

### 9.3 Walk through the COO / cultural-fit query: why 0.0 at first, and what brought it to 0.50?

**The query:** "I am looking for a COO for my company in China and I want to see if they are culturally a right fit for our company... complete in about an hour". It has 6 relevant items.

**Why it scored 0.0 at first:** the rule-based parser had three blind spots:

- it didn't recognise "COO"
- "right fit" and "culture" weren't behavioural terms
- "about an hour" wasn't parsed

So the parse had no skills, no types and no duration. With no boosts, the results came from vector search alone on a vague query: 0 of 6 hits.

**The fixes:**

1. **C-suite titles:** role "coo", and add "leadership" plus Personality & Behavior.
2. **Cultural-fit terms:** "right fit" matches directly, and "culturally" stem-matches "culture". That adds Personality & Behavior and Competencies.
3. **Duration:** "about an hour" becomes a maximum of 60.

**Result:** personality and competency tests now get +0.20 per matching type, plus the leadership keyword boost. 3 of 6 relevant items reached the top 10, giving **0.50**. Gemini would probably have handled this query anyway; the fix improved the fallback, which is now the only path.

### 9.4 Which failure types are still left, and what would you try next?

**Remaining failure types:**

1. **Pre-packaged solutions** (11 of 65): can't be fixed without changing the scope.
2. **SHL-specific pairings** in the labels, such as general aptitude or personality bundles for a role, which semantics don't imply.
3. **Long job descriptions:** still noisy, and the original text is truncated to about 512 tokens when embedded.
4. **Parser gaps:** plurals, multi-word roles, the "senior" role problem, and false-positive skills ("it", "data").
5. **Substring keyword noise.**
6. **Balance crowding out** pure aptitude tests.
7. **Unknown or zero durations** passing the filter.

**Next steps, in order:**

1. word-boundary matching and a fix for role extraction (cheap)
2. Reciprocal Rank Fusion to merge the sources
3. a cross-encoder re-ranking the top 50
4. weights learned with leave-one-out cross-validation
5. LLM query expansion
6. adding pre-packaged solutions

## 10. Live ground-truth walkthrough (skipped)

<p class="skip">Questions 10 to 10.6.1 require running the pipeline, so they are skipped. To prepare: run <code>python -m eval.evaluate --verbose</code> with logging at INFO. Then, for train query 1 and one long job description, note down the parsed query, the candidates from each retrieval path, the items the filter removed, and the top-10 scores.</p>

## 11. Deployment

### 11. How is it deployed, and on which port and platform?

**Docker image:**

1. `python:3.10-slim` plus build-essential and cmake (to compile llama-cpp-python)
2. install the requirements and copy the code
3. build the embeddings at image build time if the GGUF file is present
4. `EXPOSE 8080`, then run `uvicorn api.main:app --host 0.0.0.0 --port 8080`

**Live setup** (per the docs): a GCP Compute Engine VM (4 vCPU, 15 GB RAM, Debian 11), with the API on port **8000** and Streamlit on **8501**. The frontend finds the API through `API_URL`, which defaults to localhost:8000.

**Be ready to explain:**

- **Ports:** the Dockerfile uses 8080 (the Cloud Run convention) but the VM served the API on 8000.
- **Platform:** the docs also mention Cloud Run and Render, but the final setup was the VM.
- **Missing files:** the model file and `.npy` are gitignored, so a fresh clone needs a download step.

### 11.1 What happens on a cold start, and how do you preload?

**At startup:** the FastAPI lifespan hook calls `search._ensure_loaded()`. That loads `catalogue.json` (~180 KB) and `embeddings.npy` (389 × 768 float32, ~1.2 MB) into memory.

**Not preloaded:**

- **The embedding model.** `get_model()` is lazy, so the first request that embeds a query loads the ~140 MB GGUF, which takes a few seconds.
- **The Gemini client.** It's also created lazily, on the first request.

**If `embeddings.npy` is missing:**

- The lifespan hook logs the error and the API still starts.
- Every `/recommend` call then returns 500.
- `/health` still returns "healthy", so the health check doesn't reflect whether the service is actually ready.

**Improvements:**

- in the lifespan hook, call `get_model()` and run a warm-up `embed_query("warmup")`
- add a readiness endpoint that checks the model and embeddings are loaded
- fail fast when files are missing
