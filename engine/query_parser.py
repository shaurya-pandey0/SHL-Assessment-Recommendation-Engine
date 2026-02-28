"""
Query parser for extracting structured constraints from natural language queries.

Uses rule-based extraction (regex + keyword matching) — no LLM needed.
This ensures fast, deterministic, testable parsing that works offline.

Extracts:
  - max_duration: maximum assessment duration in minutes
  - test_types: list of desired test type categories
  - remote_required: whether remote testing is required
  - keywords: remaining important keywords for context
"""

import re
from typing import Optional


# Test type keywords and their canonical names
TEST_TYPE_KEYWORDS = {
    # Knowledge & Skills
    "knowledge": "Knowledge & Skills",
    "skills": "Knowledge & Skills",
    "programming": "Knowledge & Skills",
    "coding": "Knowledge & Skills",
    "technical": "Knowledge & Skills",

    # Cognitive / Ability & Aptitude
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

    # Personality & Behavior
    "personality": "Personality & Behavior",
    "behavior": "Personality & Behavior",
    "behavioural": "Personality & Behavior",
    "behavioral": "Personality & Behavior",
    "leadership": "Personality & Behavior",
    "motivation": "Personality & Behavior",

    # Competencies
    "competency": "Competencies",
    "competencies": "Competencies",

    # Biodata & Situational Judgement
    "biodata": "Biodata & Situational Judgement",
    "situational": "Biodata & Situational Judgement",
    "judgement": "Biodata & Situational Judgement",
    "judgment": "Biodata & Situational Judgement",

    # Simulations
    "simulation": "Simulations",
    "simulations": "Simulations",

    # Development & 360
    "development": "Development & 360",
    "360": "Development & 360",

    # Assessment Exercises
    "exercise": "Assessment Exercises",
    "exercises": "Assessment Exercises",
    "role-play": "Assessment Exercises",
    "roleplay": "Assessment Exercises",
    "in-basket": "Assessment Exercises",
    "inbox": "Assessment Exercises",
}

# Duration extraction patterns
DURATION_PATTERNS = [
    # "under 30 minutes", "less than 30 min", "max 40 mins"
    r"(?:under|less\s+than|max(?:imum)?|within|up\s+to|no\s+more\s+than)\s+(\d+)\s*(?:min(?:ute)?s?|mins?)",
    # "30 minutes or less", "40 min max"
    r"(\d+)\s*(?:min(?:ute)?s?|mins?)\s+(?:or\s+less|max(?:imum)?|or\s+under)",
    # "duration: 30", "time: 30 minutes", "30-minute"
    r"(?:duration|time|length)[\s:]*(\d+)\s*(?:min(?:ute)?s?|mins?)?",
    r"(\d+)[-\s]?min(?:ute)?s?\b",
]

# Remote testing keywords
REMOTE_KEYWORDS = [
    "remote", "online", "virtual", "proctored",
    "work from home", "wfh", "distance",
]

# Words to strip from keyword extraction (stop words + parsed constraints)
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "might", "can", "may", "shall", "must", "need", "want",
    "looking", "for", "to", "in", "on", "at", "by", "with", "from",
    "of", "and", "or", "but", "not", "no", "that", "this", "it",
    "i", "we", "they", "me", "us", "my", "our", "your", "their",
    "test", "tests", "assessment", "assessments", "exam", "exams",
    "under", "less", "than", "max", "maximum", "within", "up",
    "minutes", "minute", "mins", "min", "duration", "time", "length",
    "find", "search", "get", "give", "show", "provide", "recommend",
    "also", "about", "like", "such", "some", "any", "please",
}


def parse_query(query: str) -> dict:
    """
    Parse a natural language query into structured constraints.

    Args:
        query: Natural language query string.

    Returns:
        Dictionary with keys:
            - max_duration: int or None
            - test_types: list of test type strings
            - remote_required: bool or None
            - keywords: list of remaining relevant keywords
    """
    query_lower = query.lower().strip()

    result = {
        "max_duration": _extract_duration(query_lower),
        "test_types": _extract_test_types(query_lower),
        "remote_required": _extract_remote(query_lower),
        "keywords": _extract_keywords(query_lower),
    }

    return result


def _extract_duration(query: str) -> Optional[int]:
    """Extract maximum duration constraint from query."""
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
    found_types = set()

    # Tokenize and check each word
    words = re.findall(r"[\w'-]+", query)
    for word in words:
        word_lower = word.lower()
        if word_lower in TEST_TYPE_KEYWORDS:
            found_types.add(TEST_TYPE_KEYWORDS[word_lower])

    # Also check multi-word patterns
    for phrase, test_type in TEST_TYPE_KEYWORDS.items():
        if phrase in query:
            found_types.add(test_type)

    return sorted(found_types)


def _extract_remote(query: str) -> Optional[bool]:
    """Check if remote testing is required."""
    for keyword in REMOTE_KEYWORDS:
        if keyword in query:
            return True
    return None


def _extract_keywords(query: str) -> list[str]:
    """Extract remaining meaningful keywords from the query."""
    words = re.findall(r"[\w'-]+", query.lower())

    # Filter out stop words, test type keywords, and short words
    keywords = []
    for word in words:
        if (
            word not in STOP_WORDS
            and word not in TEST_TYPE_KEYWORDS
            and len(word) > 2
            and not word.isdigit()
        ):
            keywords.append(word)

    return keywords


if __name__ == "__main__":
    # Test with sample queries
    test_queries = [
        "Need a cognitive test for analysts, max 40 minutes",
        "Python programming assessment under 30 minutes",
        "Looking for remote personality assessment for leaders",
        "Java developer test, 45 min max, online",
        "Graduate numerical reasoning aptitude test",
        "Customer service simulation exercise, less than 60 minutes",
    ]

    for query in test_queries:
        result = parse_query(query)
        print(f"\nQuery: '{query}'")
        print(f"  Duration:  {result['max_duration']}")
        print(f"  Types:     {result['test_types']}")
        print(f"  Remote:    {result['remote_required']}")
        print(f"  Keywords:  {result['keywords']}")
