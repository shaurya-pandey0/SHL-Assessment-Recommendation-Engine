"""
Re-ranker and test-type balancer for SHL recommendations.

After vector search returns candidates, this module:
1. Applies hard filters (duration constraint)
2. Boosts scores for matching test types (+0.20 per matching type)
3. Boosts scores for keyword matches in assessment name/description
4. Balances technical (K/S) and behavioral (P/C/B) assessments
   when the query requires both — guarantees minimum 2 from minority group
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Categories for balancing
TECHNICAL_TYPES = {"Knowledge & Skills", "Simulations"}
BEHAVIORAL_TYPES = {"Personality & Behavior", "Competencies", "Biodata & Situational Judgement"}

# Generic terms that match too many assessments — exclude from keyword boost
GENERIC_SKILLS = {
    "engineering", "development", "software", "technology", "technical",
    "testing", "it", "systems", "data", "management", "infrastructure",
    "programming", "coding", "cloud", "security", "analytics",
}


def rerank(
    candidates: list[dict],
    parsed_query: dict,
    top_k: int = 10,
) -> list[dict]:
    """
    Re-rank candidates based on parsed query constraints.

    Args:
        candidates: Candidates from vector + keyword search, each with 'score' field.
        parsed_query: Output from query_parser.parse_query().
        top_k: Number of results to return.

    Returns:
        Re-ranked list of top_k assessments.
    """
    # STEP 1: Apply hard filters
    filtered = list(candidates)  # copy

    max_duration = parsed_query.get("max_duration")
    if max_duration is not None:
        filtered = [
            c for c in filtered
            if c.get("duration") is None  # keep if duration unknown
            or c["duration"] <= max_duration
        ]

    # STEP 2: Boost matching test_types
    needed_types = set(parsed_query.get("test_types_needed", []))
    for c in filtered:
        candidate_types = set(c.get("test_type", []))
        type_overlap = candidate_types & needed_types
        if type_overlap:
            c["score"] = c.get("score", 0) + 0.20 * len(type_overlap)

    # STEP 2.5: Keyword matching boost — only use SPECIFIC skills (not generic ones)
    skills = (
        parsed_query.get("skills_technical", [])
        + parsed_query.get("skills_behavioral", [])
    )
    # Filter out generic terms that would match too many assessments
    specific_skills = [s for s in skills if s.lower() not in GENERIC_SKILLS]
    if specific_skills:
        for c in filtered:
            text = (c.get("name", "") + " " + c.get("description", "")).lower()
            keyword_hits = sum(1 for s in specific_skills if s.lower() in text)
            name_lower = c.get("name", "").lower()
            name_hits = sum(1 for s in specific_skills if s.lower() in name_lower)
            if keyword_hits > 0:
                # Strong boost: name matches are the strongest signal
                c["score"] = c.get("score", 0) + 0.20 * keyword_hits + 0.15 * name_hits

    # STEP 3: Balance if needed
    if parsed_query.get("requires_balance", False):
        results = balance_test_types(filtered, top_k)
    else:
        results = sorted(filtered, key=lambda x: x.get("score", 0), reverse=True)[:top_k]

    return results


def balance_test_types(candidates: list[dict], top_k: int) -> list[dict]:
    """
    Ensure a mix of technical (K/S) and behavioral (P/C/B) assessments.

    Guarantees minimum 2 slots for the minority group, rest goes to
    the majority group by score. This prevents over-allocation to the
    minority when the query is predominantly one type.
    """
    technical = []
    behavioral = []
    other = []

    for c in candidates:
        c_types = set(c.get("test_type", []))
        if c_types & TECHNICAL_TYPES:
            technical.append(c)
        elif c_types & BEHAVIORAL_TYPES:
            behavioral.append(c)
        else:
            other.append(c)

    # Sort each group by score
    technical.sort(key=lambda x: x.get("score", 0), reverse=True)
    behavioral.sort(key=lambda x: x.get("score", 0), reverse=True)
    other.sort(key=lambda x: x.get("score", 0), reverse=True)

    # Handle empty groups
    if not behavioral:
        return (technical + other)[:top_k]
    if not technical:
        return (behavioral + other)[:top_k]

    # Guarantee minimum 2 from minority group, rest from majority by score
    MIN_MINORITY = 2
    if len(technical) >= len(behavioral):
        # Technical is majority
        behav_count = min(len(behavioral), max(MIN_MINORITY, top_k // 4))
        tech_count = min(len(technical), top_k - behav_count)
    else:
        # Behavioral is majority
        tech_count = min(len(technical), max(MIN_MINORITY, top_k // 4))
        behav_count = min(len(behavioral), top_k - tech_count)

    result = technical[:tech_count] + behavioral[:behav_count]

    # Fill remaining slots from unused candidates by score
    used_urls = {r["url"] for r in result}
    remaining = [c for c in candidates if c["url"] not in used_urls]
    remaining.sort(key=lambda x: x.get("score", 0), reverse=True)
    result.extend(remaining[:top_k - len(result)])

    return result[:top_k]
