"""
Re-ranker and test-type balancer for SHL recommendations.

After vector search returns top-20 candidates, this module:
1. Applies hard filters (duration constraint)
2. Boosts scores for matching test types (+0.15 per matching type)
3. Balances technical (K/S) and behavioral (P/C/B) assessments
   when the query requires both
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Categories for balancing
TECHNICAL_TYPES = {"Knowledge & Skills", "Simulations"}
BEHAVIORAL_TYPES = {"Personality & Behavior", "Competencies", "Biodata & Situational Judgement"}


def rerank(
    candidates: list[dict],
    parsed_query: dict,
    top_k: int = 10,
) -> list[dict]:
    """
    Re-rank candidates based on parsed query constraints.

    Args:
        candidates: Top-20 from vector search, each with 'score' field.
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
        c = c  # in-place modification
        candidate_types = set(c.get("test_type", []))
        type_overlap = candidate_types & needed_types
        if type_overlap:
            c["score"] = c.get("score", 0) + 0.15 * len(type_overlap)

    # STEP 3: Balance if needed
    if parsed_query.get("requires_balance", False):
        results = balance_test_types(filtered, top_k)
    else:
        results = sorted(filtered, key=lambda x: x.get("score", 0), reverse=True)[:top_k]

    return results


def balance_test_types(candidates: list[dict], top_k: int) -> list[dict]:
    """
    Ensure a mix of technical (K/S) and behavioral (P/C/B) assessments.

    Interleaves roughly 50/50, biasing toward the larger group.
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

    # Interleave: roughly half and half
    tech_count = min(len(technical), top_k // 2 + (1 if len(technical) > len(behavioral) else 0))
    behav_count = min(len(behavioral), top_k - tech_count)

    result = technical[:tech_count] + behavioral[:behav_count]

    # Fill remaining slots from unused candidates by score
    used_urls = {r["url"] for r in result}
    remaining = [c for c in candidates if c["url"] not in used_urls]
    remaining.sort(key=lambda x: x.get("score", 0), reverse=True)
    result.extend(remaining[:top_k - len(result)])

    return result[:top_k]
