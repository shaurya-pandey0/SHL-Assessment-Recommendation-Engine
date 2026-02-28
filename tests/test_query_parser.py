"""
Tests for the query parser.

Tests cover:
  - Duration extraction from various formats
  - Test type keyword detection
  - Remote testing detection
  - Keyword extraction
  - Full parse_query integration
"""

import pytest

from engine.query_parser import (
    parse_query,
    _extract_duration,
    _extract_test_types,
    _extract_remote,
    _extract_keywords,
)


class TestExtractDuration:
    """Tests for duration extraction from queries."""

    def test_under_N_minutes(self):
        assert _extract_duration("under 30 minutes") == 30

    def test_less_than_N_min(self):
        assert _extract_duration("less than 45 min") == 45

    def test_max_N_minutes(self):
        assert _extract_duration("max 40 minutes") == 40

    def test_maximum_N_mins(self):
        assert _extract_duration("maximum 60 mins") == 60

    def test_within_N_minutes(self):
        assert _extract_duration("within 25 minutes") == 25

    def test_N_minutes_or_less(self):
        assert _extract_duration("30 minutes or less") == 30

    def test_N_min_max(self):
        assert _extract_duration("45 min max") == 45

    def test_N_minute_hyphenated(self):
        assert _extract_duration("30-minute assessment") == 30

    def test_duration_with_label(self):
        assert _extract_duration("duration: 30") == 30

    def test_no_duration(self):
        assert _extract_duration("python developer assessment") is None

    def test_empty_string(self):
        assert _extract_duration("") is None

    def test_up_to_N_minutes(self):
        assert _extract_duration("up to 50 minutes") == 50


class TestExtractTestTypes:
    """Tests for test type extraction."""

    def test_cognitive(self):
        result = _extract_test_types("cognitive test for analysts")
        assert "Ability & Aptitude" in result

    def test_personality(self):
        result = _extract_test_types("personality assessment")
        assert "Personality & Behavior" in result

    def test_programming_maps_to_knowledge(self):
        result = _extract_test_types("programming assessment")
        assert "Knowledge & Skills" in result

    def test_skills_keyword(self):
        result = _extract_test_types("skills test for candidates")
        assert "Knowledge & Skills" in result

    def test_simulation(self):
        result = _extract_test_types("simulation exercise for customer service")
        assert "Simulations" in result

    def test_multiple_types(self):
        result = _extract_test_types("cognitive and personality assessment")
        assert "Ability & Aptitude" in result
        assert "Personality & Behavior" in result

    def test_no_type_found(self):
        result = _extract_test_types("general test for candidates")
        assert result == []

    def test_aptitude(self):
        result = _extract_test_types("aptitude test")
        assert "Ability & Aptitude" in result

    def test_reasoning(self):
        result = _extract_test_types("reasoning assessment")
        assert "Ability & Aptitude" in result

    def test_situational_judgement(self):
        result = _extract_test_types("situational judgement test")
        assert "Biodata & Situational Judgement" in result


class TestExtractRemote:
    """Tests for remote testing requirement extraction."""

    def test_remote_keyword(self):
        assert _extract_remote("remote testing needed") is True

    def test_online_keyword(self):
        assert _extract_remote("online assessment") is True

    def test_virtual_keyword(self):
        assert _extract_remote("virtual proctored test") is True

    def test_no_remote(self):
        assert _extract_remote("in-person assessment") is None

    def test_proctored_keyword(self):
        assert _extract_remote("proctored examination") is True

    def test_empty_string(self):
        assert _extract_remote("") is None


class TestExtractKeywords:
    """Tests for keyword extraction."""

    def test_extracts_meaningful_words(self):
        keywords = _extract_keywords("python developer assessment")
        assert "python" in keywords
        assert "developer" in keywords

    def test_removes_stop_words(self):
        keywords = _extract_keywords("looking for a test to find python developers")
        assert "looking" not in keywords
        assert "for" not in keywords
        assert "python" in keywords
        assert "developers" in keywords

    def test_removes_short_words(self):
        keywords = _extract_keywords("do it now for me")
        # All are stop words or short
        assert len(keywords) == 0

    def test_removes_numbers(self):
        keywords = _extract_keywords("assessment under 30 minutes")
        assert "30" not in keywords


class TestParseQueryIntegration:
    """Integration tests for the full parse_query function."""

    def test_full_query_parsing(self):
        result = parse_query("Need a cognitive test for analysts, max 40 minutes")
        assert result["max_duration"] == 40
        assert "Ability & Aptitude" in result["test_types"]
        assert "analysts" in result["keywords"]
        assert result["remote_required"] is None

    def test_python_query(self):
        result = parse_query("Python programming assessment under 30 minutes")
        assert result["max_duration"] == 30
        assert "Knowledge & Skills" in result["test_types"]
        assert "python" in result["keywords"]

    def test_remote_personality_query(self):
        result = parse_query("Looking for remote personality assessment for leaders")
        assert result["remote_required"] is True
        assert "Personality & Behavior" in result["test_types"]
        assert "leaders" in result["keywords"]

    def test_complex_query(self):
        result = parse_query(
            "Java developer test, 45 min max, online, need cognitive skills"
        )
        assert result["max_duration"] == 45
        assert result["remote_required"] is True
        assert "java" in result["keywords"]

    def test_empty_query(self):
        result = parse_query("")
        assert result["max_duration"] is None
        assert result["test_types"] == []
        assert result["remote_required"] is None
        assert result["keywords"] == []

    def test_ambiguous_query(self):
        result = parse_query("I need a 30 minute test")
        assert result["max_duration"] == 30
        # No specific test type should be extracted
        # (this is a known limitation the plan acknowledges)
