"""
Tests for the query parser.

Tests:
  - Duration extraction
  - Test type detection
  - Technical/behavioral skill extraction
  - Balance detection
  - Full parse_query integration
"""

import pytest

from engine.query_parser import (
    parse_query,
    _extract_duration,
    _extract_test_types,
    _extract_skills,
    TECHNICAL_SKILLS,
    BEHAVIORAL_SKILLS,
)


class TestExtractDuration:
    def test_under_N_minutes(self):
        assert _extract_duration("under 30 minutes") == 30

    def test_less_than(self):
        assert _extract_duration("less than 45 min") == 45

    def test_max(self):
        assert _extract_duration("max 40 minutes") == 40

    def test_N_min_max(self):
        assert _extract_duration("45 min max") == 45

    def test_no_duration(self):
        assert _extract_duration("python developer assessment") is None

    def test_up_to(self):
        assert _extract_duration("up to 50 minutes") == 50

    def test_hyphenated(self):
        assert _extract_duration("30-minute assessment") == 30


class TestExtractTestTypes:
    def test_cognitive(self):
        result = _extract_test_types("cognitive test")
        assert "Ability & Aptitude" in result

    def test_personality(self):
        result = _extract_test_types("personality assessment")
        assert "Personality & Behavior" in result

    def test_programming(self):
        result = _extract_test_types("programming assessment")
        assert "Knowledge & Skills" in result

    def test_simulation(self):
        result = _extract_test_types("simulation exercise")
        assert "Simulations" in result

    def test_multiple(self):
        result = _extract_test_types("cognitive and personality")
        assert "Ability & Aptitude" in result
        assert "Personality & Behavior" in result

    def test_none(self):
        result = _extract_test_types("general test")
        assert result == []


class TestExtractSkills:
    def test_technical_python(self):
        result = _extract_skills("python developer", TECHNICAL_SKILLS)
        assert "python" in result

    def test_technical_java(self):
        result = _extract_skills("java programming", TECHNICAL_SKILLS)
        assert "java" in result

    def test_behavioral_leadership(self):
        result = _extract_skills("leadership assessment", BEHAVIORAL_SKILLS)
        assert "leadership" in result

    def test_behavioral_collaboration(self):
        result = _extract_skills("team collaboration skills", BEHAVIORAL_SKILLS)
        assert "collaboration" in result

    def test_no_skills(self):
        result = _extract_skills("general test", TECHNICAL_SKILLS)
        assert result == []


class TestParseQueryIntegration:
    def test_technical_only(self):
        result = parse_query("Python programming assessment under 30 minutes")
        assert result["max_duration"] == 30
        assert "python" in result["skills_technical"]
        assert result["requires_balance"] is False
        assert "Knowledge & Skills" in result["test_types_needed"]

    def test_behavioral_only(self):
        result = parse_query("leadership personality assessment")
        assert "leadership" in result["skills_behavioral"]
        assert "Personality & Behavior" in result["test_types_needed"]

    def test_mixed_requires_balance(self):
        result = parse_query("Java developer who collaborates with external teams")
        assert "java" in result["skills_technical"]
        assert "collaboration" in result["skills_behavioral"]
        assert result["requires_balance"] is True

    def test_empty_query(self):
        result = parse_query("")
        assert result["max_duration"] is None
        assert result["test_types_needed"] == []
        assert result["requires_balance"] is False

    def test_complex_query(self):
        result = parse_query(
            "Senior software engineer with leadership and communication, max 45 min"
        )
        assert result["max_duration"] == 45
        assert "software" in result["skills_technical"]
        assert "leadership" in result["skills_behavioral"]
        assert result["requires_balance"] is True

    def test_cognitive_query(self):
        result = parse_query("cognitive ability test for analysts")
        assert "Ability & Aptitude" in result["test_types_needed"]
