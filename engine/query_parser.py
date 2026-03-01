"""
Query parser for extracting structured intent from natural language queries.

Two modes:
  1. Gemini-powered (Phase 3): Uses Google Gemini 1.5 Flash for intelligent
     extraction. Set GEMINI_API_KEY env var to enable.
  2. Rule-based fallback: Fast, deterministic, no API needed.

If Gemini API fails or key is missing, falls back to rule-based automatically.
The system NEVER crashes because of an LLM API failure.

Extracts:
  - job_role: detected job role
  - skills_technical: technical skill keywords
  - skills_behavioral: soft/behavioral skill keywords
  - max_duration: maximum assessment duration in minutes
  - test_types_needed: list of desired test type categories
  - requires_balance: True if BOTH technical AND behavioral skills detected
"""

import os
import re
import json
import logging
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


# ── Gemini Integration ───────────────────────────────────────────────────────

GEMINI_PROMPT = '''Analyze this hiring/assessment query and extract structured information.
Return ONLY valid JSON, no markdown fences, no other text.

Query: "{query}"

Return this JSON structure:
{{
  "job_role": "string or null",
  "skills_technical": ["list of technical skills mentioned"],
  "skills_behavioral": ["list of soft/behavioral skills mentioned"],
  "max_duration": integer or null,
  "test_types_needed": ["list from: Knowledge & Skills, Personality & Behavior, Ability & Aptitude, Competencies, Biodata & Situational Judgement, Simulations, Development & 360, Assessment Exercises"],
  "requires_balance": true or false
}}

Rules:
- If the query mentions both technical skills (coding, programming, etc.)
  AND soft skills (collaboration, communication, leadership, etc.),
  set requires_balance to true and include both test types.
- If only technical: test_types_needed = ["Knowledge & Skills"]
- If only behavioral: test_types_needed = ["Personality & Behavior", "Competencies"]
- If cognitive/aptitude mentioned: include "Ability & Aptitude"
- If duration limit mentioned (e.g. "under 30 minutes"), extract as integer minutes
- max_duration should be null if no time constraint is mentioned
- job_role should be the specific role mentioned (e.g. "software developer", "analyst")
'''

_gemini_client = None


def _get_gemini_client():
    """Lazy-load Gemini client."""
    global _gemini_client
    if _gemini_client is None:
        api_key = (os.environ.get("GEMINI_API_KEY")
                   or os.environ.get("GOOGLE_API_KEY")
                   or os.environ.get("API_KEY"))
        if not api_key:
            return None
        try:
            from google import genai
            _gemini_client = genai.Client(api_key=api_key)
            logger.info("Gemini client loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load Gemini client: {e}")
            return None
    return _gemini_client


def _parse_with_gemini(query: str) -> Optional[dict]:
    """
    Parse query using Gemini API.

    Returns parsed dict on success, None on failure (triggers fallback).
    """
    client = _get_gemini_client()
    if client is None:
        return None

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=GEMINI_PROMPT.format(query=query),
            config={"temperature": 0.1, "max_output_tokens": 500},
        )

        # Extract JSON from response
        text = response.text.strip()

        # Remove markdown code fences if present
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        parsed = json.loads(text)

        # Validate required fields exist
        required = {"job_role", "skills_technical", "skills_behavioral",
                     "max_duration", "test_types_needed", "requires_balance"}
        if not required.issubset(set(parsed.keys())):
            logger.warning(f"Gemini response missing fields: {required - set(parsed.keys())}")
            return None

        # Ensure lists are actually lists
        for field in ["skills_technical", "skills_behavioral", "test_types_needed"]:
            if not isinstance(parsed.get(field), list):
                parsed[field] = []

        # Ensure max_duration is int or None
        if parsed.get("max_duration") is not None:
            try:
                parsed["max_duration"] = int(parsed["max_duration"])
            except (ValueError, TypeError):
                parsed["max_duration"] = None

        logger.info(f"Gemini parse successful: {parsed}")
        return parsed

    except Exception as e:
        logger.warning(f"Gemini parsing failed: {e}. Falling back to rule-based.")
        return None


# ── Rule-Based Parser (Fallback) ─────────────────────────────────────────────

# Technical skill keywords
TECHNICAL_SKILLS = {
    "python", "java", "javascript", "sql", "html", "css", "react",
    "angular", "node", "typescript", "c++", "c#", ".net", "ruby",
    "php", "swift", "kotlin", "go", "rust", "scala", "r",
    "aws", "azure", "gcp", "docker", "kubernetes", "linux",
    "git", "api", "rest", "graphql", "database", "data",
    "machine learning", "ai", "devops", "cloud", "networking",
    "security", "testing", "qa", "automation", "analytics",
    "programming", "coding", "software", "engineering", "development",
    "technical", "technology", "it", "systems", "infrastructure",
    "accounting", "bookkeeping", "finance", "excel", "sap",
    "autocad", "mechanical", "electrical", "cisco", "vmware",
}

# Behavioral/soft skill keywords
BEHAVIORAL_SKILLS = {
    "leadership", "communication", "collaboration", "teamwork",
    "management", "interpersonal", "negotiation", "presentation",
    "problem solving", "critical thinking", "decision making",
    "emotional intelligence", "empathy", "adaptability", "flexibility",
    "creativity", "innovation", "strategic", "planning", "coaching",
    "mentoring", "conflict resolution", "customer service",
    "stakeholder", "relationship", "influence", "motivation",
    "personality", "behavioral", "behavioural", "soft skills",
    "people skills", "cultural fit", "culture", "work style", "attitude",
    "right fit", "team fit", "organizational fit",
}

# Test type keywords → canonical names
TEST_TYPE_KEYWORDS = {
    "knowledge": "Knowledge & Skills",
    "skills": "Knowledge & Skills",
    "programming": "Knowledge & Skills",
    "coding": "Knowledge & Skills",
    "technical": "Knowledge & Skills",
    "cognitive": "Ability & Aptitude",
    "aptitude": "Ability & Aptitude",
    "ability": "Ability & Aptitude",
    "reasoning": "Ability & Aptitude",
    "numerical": "Ability & Aptitude",
    "verbal": "Ability & Aptitude",
    "logical": "Ability & Aptitude",
    "analytical": "Ability & Aptitude",
    "inductive": "Ability & Aptitude",
    "deductive": "Ability & Aptitude",
    "personality": "Personality & Behavior",
    "behavior": "Personality & Behavior",
    "behavioural": "Personality & Behavior",
    "behavioral": "Personality & Behavior",
    "leadership": "Personality & Behavior",
    "motivation": "Personality & Behavior",
    "competency": "Competencies",
    "competencies": "Competencies",
    "biodata": "Biodata & Situational Judgement",
    "situational": "Biodata & Situational Judgement",
    "judgement": "Biodata & Situational Judgement",
    "judgment": "Biodata & Situational Judgement",
    "simulation": "Simulations",
    "simulations": "Simulations",
    "development": "Development & 360",
    "360": "Development & 360",
    "exercise": "Assessment Exercises",
    "exercises": "Assessment Exercises",
}

DURATION_PATTERNS = [
    r"(?:under|less\s+than|max(?:imum)?|within|up\s+to|no\s+more\s+than)\s+(\d+)\s*(?:min(?:ute)?s?|mins?)",
    r"(\d+)\s*(?:min(?:ute)?s?|mins?)\s+(?:or\s+less|max(?:imum)?|or\s+under)",
    r"(?:duration|time|length)[\s:]*(\d+)\s*(?:min(?:ute)?s?|mins?)?",
    r"(\d+)[-\s]?min(?:ute)?s?\b",
    r"(?:about|around|approximately)\s+(?:an?\s+)?hour",  # "about an hour" → 60 min
    r"(\d+)[-\s]?hours?\b",  # "1-2 hours" → captures first number
]

JOB_ROLES = {
    "developer", "engineer", "analyst", "manager", "director",
    "designer", "architect", "administrator", "consultant",
    "specialist", "coordinator", "executive", "supervisor",
    "assistant", "officer", "representative", "agent",
    "technician", "operator", "clerk", "accountant",
    "graduate", "intern", "junior", "senior", "mid-level",
    "entry-level", "professional",
    # C-suite titles
    "coo", "ceo", "cto", "cfo", "cmo", "cio", "vp",
}

# Executive titles that imply leadership/personality assessment needs
EXECUTIVE_TITLES = {"coo", "ceo", "cto", "cfo", "cmo", "cio", "vp", "director", "executive"}


def _parse_rule_based(query: str) -> dict:
    """Rule-based fallback parser. Fast, deterministic, no API needed."""
    query_lower = query.lower().strip()

    skills_tech = _extract_skills(query_lower, TECHNICAL_SKILLS)
    skills_behav = _extract_skills(query_lower, BEHAVIORAL_SKILLS)
    test_types = _extract_test_types(query_lower)

    if skills_tech and "Knowledge & Skills" not in test_types:
        test_types.append("Knowledge & Skills")
    if skills_behav and "Personality & Behavior" not in test_types:
        test_types.append("Personality & Behavior")
    if skills_behav and "Competencies" not in test_types:
        test_types.append("Competencies")

    # Detect executive titles → auto-add leadership/personality context
    job_role = _extract_job_role(query_lower)
    words = set(re.findall(r"[\w-]+", query_lower))
    if words & EXECUTIVE_TITLES:
        if "leadership" not in skills_behav:
            skills_behav.append("leadership")
        if "Personality & Behavior" not in test_types:
            test_types.append("Personality & Behavior")

    requires_balance = bool(skills_tech) and bool(skills_behav)

    return {
        "job_role": job_role,
        "skills_technical": skills_tech,
        "skills_behavioral": skills_behav,
        "max_duration": _extract_duration(query_lower),
        "test_types_needed": sorted(set(test_types)),
        "requires_balance": requires_balance,
    }


# ── Public API ───────────────────────────────────────────────────────────────

def parse_query(query: str) -> dict:
    """
    Parse a natural language query into structured constraints.

    Tries Gemini first (if API key available), falls back to rule-based.

    Returns dict with:
        - job_role: str or None
        - skills_technical: list[str]
        - skills_behavioral: list[str]
        - max_duration: int or None
        - test_types_needed: list[str]
        - requires_balance: bool
    """
    # Try Gemini first
    result = _parse_with_gemini(query)
    if result is not None:
        return result

    # Fallback to rule-based
    return _parse_rule_based(query)


# ── Helper Functions ─────────────────────────────────────────────────────────

def _extract_duration(query: str) -> Optional[int]:
    """Extract maximum duration constraint."""
    # Handle "about an hour" / "about X hours" first
    if re.search(r"(?:about|around|approximately)\s+(?:an?\s+)?hour\b", query, re.IGNORECASE):
        return 60
    # Handle "X hours" → convert to minutes
    hours_match = re.search(r"(\d+)(?:\s*-\s*\d+)?\s*hours?\b", query, re.IGNORECASE)
    if hours_match:
        return int(hours_match.group(1)) * 60

    for pattern in DURATION_PATTERNS:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                continue
    return None


def _extract_test_types(query: str) -> list[str]:
    """Extract desired test types from query keywords."""
    found = set()
    words = re.findall(r"[\w'-]+", query)
    for word in words:
        if word.lower() in TEST_TYPE_KEYWORDS:
            found.add(TEST_TYPE_KEYWORDS[word.lower()])
    return sorted(found)


def _extract_skills(query: str, skill_set: set) -> list[str]:
    """Extract skills from query that match the given skill set."""
    found = []
    query_lower = query.lower()

    # Check multi-word skills first (longer matches)
    for skill in sorted(skill_set, key=len, reverse=True):
        if " " in skill and skill in query_lower:
            found.append(skill)

    # Then single-word skills with fuzzy stem matching
    words = set(re.findall(r"[\w#+.-]+", query_lower))
    for word in words:
        if word in skill_set and word not in found:
            found.append(word)
        elif len(word) >= 5:
            # Stem-like matching: check if word shares a root with any skill
            for skill in skill_set:
                if " " in skill or len(skill) < 5:
                    continue
                prefix_len = min(len(word), len(skill)) - 2
                if prefix_len < 5:
                    continue
                if (word[:prefix_len] == skill[:prefix_len]
                    and skill not in found):
                    found.append(skill)
                    break

    return sorted(found)


def _extract_job_role(query: str) -> Optional[str]:
    """Extract job role from query."""
    words = re.findall(r"[\w-]+", query.lower())
    for i, word in enumerate(words):
        if word in JOB_ROLES:
            if i > 0 and words[i-1] not in {"a", "an", "the", "for", "as"}:
                return f"{words[i-1]} {word}"
            return word
    return None
