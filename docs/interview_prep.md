# 🎯 SHL Interview Prep — Your Cheat Sheet

> **What this is:** Everything you need to confidently walk into the SHL first-round interview and own every question about your submission. Read this once the night before. Skim it again in the morning. You're set.

---

## 🧠 The 30-Second Elevator Pitch

> *"I built a retrieval-based recommendation system that takes a natural language query or job description URL and returns up to 10 relevant SHL assessments. I scraped 389 assessments from SHL's catalogue using Selenium, embedded them with Sentence-BERT, and built a pipeline that does vector search, structured query parsing via Gemini with a rule-based fallback, and test-type-aware re-ranking with balanced interleaving. The system is served via FastAPI with a Streamlit frontend."*

Say this smoothly. Practice it once. Don't memorize it word-for-word — just hit these beats:
**scrape → embed → search → parse → rerank → serve**

---

## 🔥 What They're Actually Evaluating

From the assignment PDF, they care about **5 things**:

| What They Want | Where You Deliver |
|---|---|
| **Solution Approach** — clear pipeline, modular code | `engine/` — 5 clean modules, each with one job |
| **Data Pipeline** — scraping, parsing, structured storage | `scraper/scrape_catalogue.py` → `catalogue.json` (389 assessments) |
| **LLM Integration** — not just keyword matching | Gemini 2.0 Flash for query parsing + Sentence-BERT for embeddings |
| **Evaluation** — measurable, not vibes | Recall@10, MAP@10, per-query breakdown |
| **Performance & Balance** — mixed queries handled | Re-ranker with type boosting + technical/behavioral interleaving |

---

## 💬 The Questions They WILL Ask (and Your Answers)

### 🏗️ Architecture & Design

**Q: Walk me through your system architecture.**

> "Six-stage pipeline: User query comes in → if it's a URL, I fetch and extract text → query parser extracts structured intent (skills, duration, test types) → Sentence-BERT embeds the query into a 384-dim vector → cosine similarity against all 389 assessment embeddings → re-ranker applies duration filtering, type boosting, and balanced interleaving → top 10 returned as JSON."

**Q: Why didn't you use a vector database like FAISS or Pinecone?**

> "389 assessments. A NumPy dot product against 389 vectors takes ~0.1 milliseconds. FAISS adds installation complexity, Pinecone adds an API dependency — both with zero performance benefit at this scale. If the catalogue grew to 100K+, I'd switch to FAISS IVF. But right now it'd be over-engineering."

*💡 This answer is gold. It shows you know FAISS exists AND you know when NOT to use it.*

**Q: Why Sentence-BERT and not OpenAI embeddings?**

> "Three reasons: (1) Runs locally, no API dependency, no rate limits, no cost. (2) `all-MiniLM-L6-v2` is only 80MB and optimized for semantic similarity — it's the standard for this task. (3) In production, you don't want your core search to fail because a third-party API is down."

**Q: Why not use an LLM to directly pick assessments?**

> "Hallucination risk. An LLM might recommend assessments that don't exist in the catalogue. By using the LLM only for query understanding and retrieval for selection, every result is guaranteed to be a real, scraped assessment. Deterministic selection > LLM hallucination."

*💡 If they push: "But couldn't you constrain the LLM output?" — say: "Yes, via function calling or structured output. But that adds latency, cost, and a failure mode. The vector search approach is faster, cheaper, and equally effective at this scale."*

---

### 🕷️ Data Pipeline

**Q: How did you scrape the data?**

> "Two-stage pipeline. Stage 1: Selenium with headless Chrome navigates the paginated listing pages — the catalogue is JavaScript-rendered, so `requests` alone can't see the table rows. I extract name, URL, remote/adaptive support, and test type codes. Stage 2: For each of the 389 URLs, I use plain `requests` + BeautifulSoup to hit the detail page and extract the description from the `og:description` meta tag and duration via regex."

**Q: Why Selenium for listings but requests for details?**

> "The listing page is a JavaScript SPA — the table rows don't exist in the initial HTML. But the detail pages are server-rendered — the meta tags and text are in the raw HTML. Using `requests` for 389 detail pages is 10x faster than spawning 389 Selenium pages."

**Q: How many assessments did you get?**

> "389 total. The assignment says 'at least 377 Individual Test Solutions.' I also included 13 pre-packaged job solutions that were on the same listing pages because they scrape together and having extra data only helps recall."

**Q: What fields did you extract?**

> "Seven fields per assessment: `name`, `url`, `remote_support`, `adaptive_support`, `test_type` (as an array of canonical names), `description`, and `duration` (int or null). 100% have descriptions, 76% have explicit durations — 92 assessments genuinely have no listed duration on SHL's site."

---

### 🤖 LLM & Embeddings

**Q: How does your query parser work?**

> "Dual-mode. Primary: Gemini 2.0 Flash with a structured prompt that returns JSON — job role, technical skills, behavioral skills, max duration, test types, and a balance flag. Fallback: if Gemini fails (rate limit, no API key, network issue), a rule-based parser using regex patterns and keyword dictionaries extracts the same fields deterministically. The system never crashes because of an LLM failure."

**Q: Why is the fallback important?**

> "Because in the real world, APIs go down. Rate limits get hit. Keys expire. If your recommendation system is unusable every time the LLM hiccups, that's a production problem. The rule-based parser isn't as nuanced, but it's always available and it works."

**Q: What text do you embed for each assessment?**

> "`{name}. Test type: {types}. {description}`. Including test type in the embedding text is deliberate — without it, the model only matches on topic similarity. With it, the vector space also separates knowledge tests from personality tests from simulations. So a query about 'personality assessment' doesn't just match on keywords — it pulls toward the personality cluster in embedding space."

**Q: Why L2 normalize the embeddings?**

> "Once you L2-normalize, cosine similarity equals dot product. So instead of computing `cos(a,b) = dot(a,b) / (norm(a) * norm(b))` every time, I just call `np.dot(embeddings, query_vec)` — one line, one operation. It's a standard trick in production retrieval systems."

---

### ⚖️ Re-ranking & Balancing

**Q: How do you handle the 'Java developer who collaborates' scenario?**

> "This is exactly what the balance system is for. The query parser detects both technical skills ('Java') and behavioral skills ('collaboration'). It sets `requires_balance = True` and `test_types_needed = ['Knowledge & Skills', 'Personality & Behavior']`. The re-ranker first boosts scores for matching types (+0.15 per match), then splits candidates into technical and behavioral groups, and interleaves them — roughly 50/50 by score within each group."

**Q: Why +0.15 for the type boost? Why not some other number?**

> "Empirically chosen. Cosine similarity scores typically range 0.2–0.8 in this system. A boost of 0.15 is large enough to reorder candidates of similar relevance (preferring the right type) but small enough that a highly relevant wrong-type result still beats an irrelevant right-type result. With training data, you could optimize this coefficient via grid search."

**Q: What happens with duration filtering?**

> "Hard filter — if `max_duration` is extracted and an assessment exceeds it, it's removed. But assessments with unknown duration (null) are kept, not removed. This is intentional: removing them would mean 24% of the catalogue is unavailable whenever any duration constraint appears."

---

### 📊 Evaluation

**Q: How did you evaluate your system?**

> "Recall@10 on the labelled training set. For each query, I run the full pipeline, extract the recommended URLs, normalize them for path variations, and compute what fraction of the ground-truth relevant URLs appear in my top 10. I also compute MAP@10 to measure ranking quality, not just presence."

**Q: What's your recall?**

> "Mean Recall@10 is approximately 0.72. There are specific failure modes — ambiguous queries without clear skill keywords produce less targeted results, and URL path mismatches between my scraper output and the ground truth can miss exact matches."

**Q: How would you improve recall?**

> "Three levers: (1) Query expansion — use the LLM to generate 2-3 paraphrases of each query and merge results. (2) Cross-encoder reranking — use a second-stage model like `ms-marco-MiniLM` that scores query-document pairs jointly, which is more precise than bi-encoder similarity. (3) Tune the type boost coefficient on training data."

---

### 🛠️ Engineering & Production

**Q: What's your cold start latency?**

> "The Sentence-BERT model takes ~3-5 seconds to load on first request. After that, all subsequent requests are 150-300ms. In the Docker deployment, I pre-load the model at startup via the lifespan handler, so the cold start only happens once."

**Q: How would you deploy this?**

> "The Dockerfile pre-downloads the model and pre-computes embeddings at build time. Deploy the container to Render or Cloud Run. Streamlit frontend goes on Streamlit Cloud. The only runtime external dependency is the optional Gemini API."

**Q: What if SHL updates their catalogue?**

> "Currently, the catalogue is a static snapshot. In production, I'd set up a scheduled re-scrape pipeline (weekly cron job), compute delta embeddings for new/changed assessments, and hot-swap the embeddings file. The API reads from disk, so you can update without downtime."

---

## ⚠️ Gotcha Questions (The Traps)

### "Why didn't you use LangChain / LlamaIndex?"

> "I evaluated them but decided they'd add abstraction without value here. LangChain is great for complex multi-step chains, but my pipeline is simple: parse → embed → search → rerank. Using LangChain would add ~50 dependencies and hide the logic behind abstractions that make debugging harder. I wanted full control over every stage."

*💡 Key insight: they list LangChain in the PDF as an option, not a requirement. Showing you know it AND chose not to use it is stronger than just using it.*

### "Is your search semantic or keyword-based?"

> "Semantic. Sentence-BERT encodes meaning, not just words. 'Programming ability' matches 'coding assessment' even though they share zero keywords. That said, the query parser does use keyword matching for structured extraction — skills, duration, test types — because those are discrete values, not semantic concepts."

### "What happens if someone enters garbage?"

> "The embedding model still produces a vector. The dot product still runs. The results won't be relevant, but the system won't crash. The top results will be whatever's closest in embedding space to the garbage input — which is a graceful failure mode."

### "What are the limitations of your approach?"

> "Four main ones: (1) Ambiguous queries with no skill keywords produce scattered results. (2) 24% of assessments have no duration data, so duration filtering isn't always precise. (3) The catalogue is a static snapshot — SHL may add/remove assessments. (4) Sentence-BERT is a bi-encoder, which is less precise than a cross-encoder for fine-grained relevance."

---

## 📐 Numbers to Know Cold

| Fact | Number |
|------|--------|
| Total assessments scraped | 389 |
| Embedding dimensions | 384 |
| Embedding model | `all-MiniLM-L6-v2` (22M params) |
| Model size | ~80MB |
| Search latency | ~0.1ms (NumPy dot product) |
| End-to-end latency | ~150-300ms |
| Type boost coefficient | +0.15 |
| Vector search candidates | top 20 → re-ranked to top 10 |
| Total tests | 90 |
| Assessments with descriptions | 100% |
| Assessments with duration | 76% (297/389) |
| Test types in catalogue | 8 (K, P, S, A, B, C, D, E) |

---

## 🎭 The Interview Energy

**DO:**
- Say "I chose X because Y" not "I used X"
- Acknowledge trade-offs: "This works at 389 scale, at 100K I'd need FAISS"
- Show you know what you *didn't* build: "A cross-encoder would improve precision"
- Be specific with numbers: "0.1ms", "384 dimensions", "389 assessments"

**DON'T:**
- Say "state of the art" or "cutting edge"
- Overclaim performance
- Get defensive about limitations — own them
- Say "I worked really hard on this" — let the work speak

---

## 🏁 Closing Statement (if they ask "anything else?")

> *"One thing I'd highlight — I designed every component with a fallback. If Gemini is down, the rule-based parser works. If the embedding model fails to load, you get a clear error. If the query is garbage, you still get results — they're just less relevant. I think production resilience matters more than peak-case performance."*

That's the kind of sentence that gets you hired.

---

*Good luck. You built the thing. Now go explain it like you own it. Because you do.* 🚀
