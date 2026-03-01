"""
Full recommendation pipeline.

Chains: query parsing → query compression → vector search →
        re-rank/balance → format output.

Query Compression:
  Long JDs (>80 words) are compressed to a short synthetic query
  by extracting: job role, technical skills, known technologies,
  behavioral keywords, and domain terms. This prevents embedding
  dilution from boilerplate JD text (benefits, DEI statements, etc.)
  and dramatically improves retrieval precision.

Output format (exact fields required by SHL):
{
    "url": str,
    "name": str,
    "adaptive_support": "Yes"/"No",
    "description": str,
    "duration": int or null,
    "remote_support": "Yes"/"No",
    "test_type": ["Knowledge & Skills", ...]  # array
}
"""

import re
import logging
from typing import Optional

from engine.search import vector_search, keyword_search
from engine.query_parser import parse_query
from engine.reranker import rerank

logger = logging.getLogger(__name__)

# ── Query Compression ────────────────────────────────────────────────────────

# Known technology / tool / domain keywords (case-insensitive matching)
KNOWN_TECHNOLOGIES = {
    # Programming languages
    "java", "javascript", "python", "sql", "html", "css", "react", "angular",
    "node.js", "typescript", "c++", "c#", ".net", "ruby", "php", "swift",
    "kotlin", "go", "rust", "scala", "r",
    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "linux", "git",
    "jenkins", "terraform", "ansible", "nginx", "apache", "devops", "ci/cd",
    # Testing
    "selenium", "manual testing", "automation testing", "api testing",
    # Data & Analytics
    "tableau", "power bi", "excel", "sap", "spss", "hadoop", "spark",
    "mongodb", "postgresql", "mysql", "redis", "kafka",
    "machine learning", "deep learning", "nlp", "computer vision",
    "data science", "data engineering", "data analysis", "data analytics",
    # Frameworks
    "django", "flask", "spring", "hibernate", "maven", "gradle", "webpack",
    "drupal", "wordpress",
    # Design
    "figma", "sketch", "photoshop", "illustrator", "autocad",
    # Tools
    "jira", "confluence", "salesforce", "matlab",
    "agile", "scrum", "cybersecurity", "penetration testing", "networking", "tcp/ip",
    # Domain terms (match assessment names in SHL catalogue)
    "marketing", "advertising", "accounting", "finance", "banking",
    "sales", "leadership", "administrative", "verbal", "numerical",
    "inductive reasoning", "email writing", "communication",
}

# Role title patterns
ROLE_PATTERNS = [
    r"\b((?:senior|junior|lead|chief|head|principal|staff|entry[- ]level|mid[- ]level)\s+)?"
    r"(software engineer|software developer|data engineer|data scientist|data analyst|"
    r"qa engineer|qa analyst|test engineer|devops engineer|sre|"
    r"product manager|project manager|program manager|marketing manager|"
    r"business analyst|system(?:s)? administrator|database administrator|"
    r"front[- ]?end developer|back[- ]?end developer|full[- ]?stack developer|"
    r"web developer|mobile developer|cloud engineer|ml engineer|ai engineer|"
    r"security analyst|network engineer|technical writer|content writer|"
    r"ux designer|ui designer|graphic designer|"
    r"sales (?:representative|executive|manager|associate)|"
    r"customer (?:support|service) (?:executive|representative|agent|specialist)|"
    r"hr (?:manager|executive|specialist|generalist)|"
    r"financial analyst|accountant|auditor|consultant|"
    r"administrative (?:assistant|professional)|receptionist|clerk|"
    r"operations manager|supply chain manager|logistics manager|"
    r"research (?:engineer|scientist|analyst)|"
    r"manager|director|supervisor|coordinator|specialist|officer|"
    r"engineer|developer|analyst|designer|administrator|executive|"
    r"assistant|representative|associate|intern)\b",
]

# Behavioral / soft-skill keywords for extraction
BEHAVIORAL_TERMS = {
    "leadership", "communication", "collaboration", "teamwork",
    "problem solving", "critical thinking", "decision making",
    "interpersonal", "negotiation", "presentation", "coaching",
    "mentoring", "people management", "stakeholder management",
    "conflict resolution", "customer service", "relationship building",
    "strategic thinking", "adaptability", "creativity", "innovation",
}

WORD_THRESHOLD = 80  # Queries longer than this are compressed


def compress_query(query: str, parsed: dict) -> str:
    """
    Compress a long JD/query into a focused synthetic query for embedding.

    For short queries (<80 words), returns the original query unchanged.
    For long queries, extracts key signals and builds a compact representation.

    Args:
        query: Raw query text (may be a full JD).
        parsed: Structured parse result from query_parser.

    Returns:
        Compressed query string for embedding.
    """
    words = query.split()
    if len(words) <= WORD_THRESHOLD:
        return query

    query_lower = query.lower()
    components = []

    # 1. Extract job role (from parsed or regex)
    if parsed.get("job_role"):
        components.append(parsed["job_role"])
    else:
        for pattern in ROLE_PATTERNS:
            match = re.search(pattern, query_lower)
            if match:
                role = match.group(0).strip()
                components.append(role)
                break

    # 2. Extract known technologies / tools
    found_tech = set()
    for tech in sorted(KNOWN_TECHNOLOGIES, key=len, reverse=True):
        if " " in tech:
            if tech in query_lower:
                found_tech.add(tech)
        else:
            # Word-boundary match for single words
            if re.search(r'\b' + re.escape(tech) + r'\b', query_lower):
                found_tech.add(tech)

    # Also include parsed technical skills
    if parsed.get("skills_technical"):
        found_tech.update(parsed["skills_technical"])

    components.extend(sorted(found_tech))

    # 3. Extract behavioral terms
    found_behav = set()
    for term in sorted(BEHAVIORAL_TERMS, key=len, reverse=True):
        if term in query_lower:
            found_behav.add(term)

    if parsed.get("skills_behavioral"):
        found_behav.update(parsed["skills_behavioral"])

    components.extend(sorted(found_behav))

    # 4. Add test type context from parsed result
    if parsed.get("test_types_needed"):
        components.extend(parsed["test_types_needed"])

    # 5. If we found very little, fall back to first 2 sentences
    if len(components) < 3:
        sentences = re.split(r'[.\n]', query)
        meaningful = [s.strip() for s in sentences if len(s.strip()) > 20][:2]
        return " ".join(meaningful) if meaningful else query

    compressed = " ".join(components)
    logger.info(f"Compressed query ({len(words)} words → {len(compressed.split())} words): {compressed[:120]}...")
    return compressed


# ── Main Pipeline ────────────────────────────────────────────────────────────

def recommend(query: str, top_k: int = 10) -> list[dict]:
    """
    Full recommendation pipeline.

    Args:
        query: Natural language query or job description text.
        top_k: Number of results to return (max 10).

    Returns:
        List of formatted recommendation dicts.
    """
    # 1. Parse query to extract structured intent
    parsed = parse_query(query)
    logger.info(f"Parsed query: {parsed}")

    # 2. Compress long queries for better embeddings
    search_query = compress_query(query, parsed)

    # 3. Vector search (dual for long queries)
    if search_query != query:
        # Long query detected — merge results from both searches
        candidates_compressed = vector_search(search_query, top_k=50)
        candidates_original = vector_search(query, top_k=50)

        # Merge by URL, taking max score
        merged = {}
        for c in candidates_compressed + candidates_original:
            url = c["url"]
            if url not in merged or c["score"] > merged[url]["score"]:
                merged[url] = c
        candidates = sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:50]
        logger.info(f"Dual search: {len(candidates_compressed)} + {len(candidates_original)} → {len(candidates)} merged")
    else:
        candidates = vector_search(search_query, top_k=50)

    # 3.25 Role-focused search: search with just the job role + test type context
    # This surfaces role-specific assessments lost in full query embedding
    role_parts = []
    if parsed.get("job_role"):
        role_parts.append(parsed["job_role"])
    if parsed.get("test_types_needed"):
        role_parts.extend(parsed["test_types_needed"])
    if parsed.get("skills_technical"):
        role_parts.extend(parsed["skills_technical"][:5])  # top 5 skills
    if parsed.get("skills_behavioral"):
        role_parts.extend(parsed["skills_behavioral"][:3])  # top 3 behavioral
    if role_parts:
        role_query = " ".join(role_parts) + " assessment"
        role_results = vector_search(role_query, top_k=20)
        existing_urls = {c["url"] for c in candidates}
        for r in role_results:
            if r["url"] not in existing_urls:
                candidates.append(r)
                existing_urls.add(r["url"])

    # 3.5 Hybrid: merge keyword search results (catches exact skill name matches
    # that semantic search may miss, e.g., "Python" → "Python (New)")
    # Extract skills from BOTH parsed result AND direct text scanning
    all_skills = list(set(
        parsed.get("skills_technical", [])
        + parsed.get("skills_behavioral", [])
    ))
    # Also scan the raw query for technology keywords (parser may miss some)
    query_lower = query.lower()
    for tech in KNOWN_TECHNOLOGIES:
        if " " in tech:
            if tech in query_lower and tech not in all_skills:
                all_skills.append(tech)
        else:
            if re.search(r'\b' + re.escape(tech) + r'\b', query_lower) and tech not in all_skills:
                all_skills.append(tech)

    # Filter out generic terms that match too many assessments
    GENERIC_TERMS = {
        "engineering", "development", "software", "technology", "technical",
        "testing", "it", "systems", "data", "management", "infrastructure",
        "programming", "coding", "cloud", "security", "analytics",
    }
    specific_skills = [s for s in all_skills if s.lower() not in GENERIC_TERMS]

    if specific_skills:
        kw_results = keyword_search(specific_skills, top_k=30)
        # Merge keyword results into candidates (don't overwrite higher vector scores)
        existing_urls = {c["url"] for c in candidates}
        added = 0
        for kw in kw_results:
            if kw["url"] not in existing_urls:
                candidates.append(kw)
                existing_urls.add(kw["url"])
                added += 1
        if added:
            logger.info(f"Keyword search added {added} new candidates from {len(kw_results)} keyword matches")

    # 4. Re-rank and balance
    results = rerank(candidates, parsed, top_k=top_k)

    # 5. Format output with exact required fields
    return [
        {
            "url": r.get("url", ""),
            "name": r.get("name", ""),
            "adaptive_support": r.get("adaptive_support", "No"),
            "description": r.get("description", ""),
            "duration": r.get("duration"),
            "remote_support": r.get("remote_support", "No"),
            "test_type": r.get("test_type", []),  # already an array
        }
        for r in results
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_queries = [
        "Python programming assessment under 30 minutes",
        "Java developer who collaborates with external teams",
        "Leadership personality assessment for managers",
        "Cognitive ability test for analysts, max 40 minutes",
        "Customer service simulation under 60 minutes",
    ]

    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: '{query}'")
        print(f"{'='*80}")
        results = recommend(query)
        for i, r in enumerate(results, 1):
            test_type = ", ".join(r["test_type"])
            print(
                f"  {i}. {r['name']:<40} "
                f"| {r['duration'] or 'N/A':>4} min "
                f"| {test_type:<25} "
                f"| remote={r['remote_support']}"
            )
