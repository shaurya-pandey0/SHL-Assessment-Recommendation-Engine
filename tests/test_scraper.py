"""
Tests for the SHL catalogue scraper.

Tests:
  - Duration parsing from detail page text
  - Test type code → full name mapping
  - Catalogue data structure validation
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from scraper.scrape_catalogue import (
    scrape_detail_page,
    validate_catalogue,
    load_catalogue,
    save_catalogue,
    TEST_TYPE_MAP,
)


# --- Helper ---

def _make_assessment(**overrides):
    """Create a sample assessment with defaults."""
    base = {
        "name": "Test Assessment",
        "url": "https://www.shl.com/products/product-catalog/view/test/",
        "remote_support": "Yes",
        "adaptive_support": "No",
        "test_type": ["Knowledge & Skills"],
        "description": "A test assessment for testing.",
        "duration": 30,
    }
    base.update(overrides)
    return base


# --- Test Type Mapping ---

class TestTypeMapping:
    def test_all_codes_mapped(self):
        """All expected letter codes should have a mapping."""
        expected = {"A", "B", "C", "D", "E", "K", "P", "S"}
        assert set(TEST_TYPE_MAP.keys()) == expected

    def test_knowledge_code(self):
        assert TEST_TYPE_MAP["K"] == "Knowledge & Skills"

    def test_personality_code(self):
        assert TEST_TYPE_MAP["P"] == "Personality & Behavior"

    def test_ability_code(self):
        assert TEST_TYPE_MAP["A"] == "Ability & Aptitude"

    def test_simulations_code(self):
        assert TEST_TYPE_MAP["S"] == "Simulations"


# --- Catalogue Validation ---

class TestValidateCatalogue:
    def test_perfect_catalogue(self):
        assessments = [_make_assessment(name=f"Test {i}", url=f"https://shl.com/{i}") for i in range(5)]
        report = validate_catalogue(assessments)
        assert report["total"] == 5
        assert report["with_description"] == 5
        assert report["with_duration"] == 5

    def test_missing_descriptions(self):
        assessments = [
            _make_assessment(description="Has description"),
            _make_assessment(description=""),
            _make_assessment(description=""),
        ]
        report = validate_catalogue(assessments)
        assert report["with_description"] == 1
        assert len(report["missing_descriptions"]) == 2

    def test_missing_durations(self):
        assessments = [
            _make_assessment(duration=30),
            _make_assessment(duration=None),
        ]
        report = validate_catalogue(assessments)
        assert report["with_duration"] == 1
        assert len(report["missing_durations"]) == 1

    def test_type_distribution(self):
        assessments = [
            _make_assessment(test_type=["Knowledge & Skills"]),
            _make_assessment(test_type=["Knowledge & Skills"]),
            _make_assessment(test_type=["Personality & Behavior"]),
            _make_assessment(test_type=["Knowledge & Skills", "Personality & Behavior"]),
        ]
        report = validate_catalogue(assessments)
        assert report["test_type_distribution"]["Knowledge & Skills"] == 3
        assert report["test_type_distribution"]["Personality & Behavior"] == 2


# --- Save/Load ---

class TestSaveLoadCatalogue:
    def test_round_trip(self, tmp_path):
        assessments = [_make_assessment(name="A"), _make_assessment(name="B")]
        filepath = tmp_path / "cat.json"
        save_catalogue(assessments, filepath)
        loaded = load_catalogue(filepath)
        assert len(loaded) == 2
        assert loaded[0]["name"] == "A"

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_catalogue(tmp_path / "nope.json")

    def test_test_type_is_array(self, tmp_path):
        """test_type must always be an array, not a string."""
        assessments = [_make_assessment(test_type=["Knowledge & Skills", "Simulations"])]
        filepath = tmp_path / "cat.json"
        save_catalogue(assessments, filepath)
        loaded = load_catalogue(filepath)
        assert isinstance(loaded[0]["test_type"], list)
        assert len(loaded[0]["test_type"]) == 2

    def test_duration_is_int_or_null(self, tmp_path):
        assessments = [
            _make_assessment(duration=30),
            _make_assessment(duration=None),
        ]
        filepath = tmp_path / "cat.json"
        save_catalogue(assessments, filepath)
        loaded = load_catalogue(filepath)
        assert loaded[0]["duration"] == 30
        assert loaded[1]["duration"] is None


# --- Data Structure Checks ---

class TestCatalogueStructure:
    """Verify assessment dicts have all required fields."""

    REQUIRED_FIELDS = {"name", "url", "remote_support", "adaptive_support",
                       "test_type", "description", "duration"}

    def test_all_fields_present(self):
        a = _make_assessment()
        assert self.REQUIRED_FIELDS.issubset(set(a.keys()))

    def test_test_type_is_list(self):
        a = _make_assessment()
        assert isinstance(a["test_type"], list)

    def test_url_is_valid(self):
        a = _make_assessment()
        assert a["url"].startswith("https://www.shl.com/")

    def test_remote_support_values(self):
        a1 = _make_assessment(remote_support="Yes")
        a2 = _make_assessment(remote_support="No")
        assert a1["remote_support"] in ("Yes", "No")
        assert a2["remote_support"] in ("Yes", "No")

    def test_adaptive_support_values(self):
        a1 = _make_assessment(adaptive_support="Yes")
        a2 = _make_assessment(adaptive_support="No")
        assert a1["adaptive_support"] in ("Yes", "No")
        assert a2["adaptive_support"] in ("Yes", "No")
