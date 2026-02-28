"""
Tests for the SHL catalogue scraper.

Tests cover:
  - Data structure and field validation
  - URL format correctness  
  - Duration parsing
  - Test type code mapping
  - Catalogue validation report
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from data.scraper import (
    _parse_duration,
    _parse_test_type_codes,
    scrape_listing_page,
    validate_catalogue,
    load_catalogue,
    save_catalogue,
    TEST_TYPE_MAP,
)


# --- Unit Tests for Helper Functions ---

class TestParseDuration:
    """Tests for duration extraction from detail page text."""

    def test_simple_number(self):
        assert _parse_duration("Approximate Completion Time in minutes = 30") == 30

    def test_large_number(self):
        assert _parse_duration("Completion Time in minutes = 120") == 120

    def test_no_number(self):
        assert _parse_duration("No time specified") is None

    def test_empty_string(self):
        assert _parse_duration("") is None

    def test_none_input(self):
        assert _parse_duration(None) is None

    def test_multiple_numbers_takes_first(self):
        # Should take the first number found
        result = _parse_duration("Time = 45 or 60 minutes")
        assert result == 45


class TestParseTestTypeCodes:
    """Tests for test type code to name conversion."""

    def test_single_code(self):
        result = _parse_test_type_codes(["K"])
        assert result == ["Knowledge & Skills"]

    def test_multiple_codes(self):
        result = _parse_test_type_codes(["C", "P", "A"])
        assert "Competencies" in result
        assert "Personality & Behavior" in result
        assert "Ability & Aptitude" in result

    def test_empty_list(self):
        assert _parse_test_type_codes([]) == []

    def test_unknown_code(self):
        result = _parse_test_type_codes(["Z"])
        assert result == ["Z"]

    def test_whitespace_handling(self):
        result = _parse_test_type_codes([" K ", " P "])
        assert "Knowledge & Skills" in result
        assert "Personality & Behavior" in result

    def test_all_known_codes(self):
        """Verify all known test type codes map correctly."""
        for code, name in TEST_TYPE_MAP.items():
            result = _parse_test_type_codes([code])
            assert result == [name], f"Code '{code}' should map to '{name}'"


# --- Tests for Catalogue Validation ---

class TestValidateCatalogue:
    """Tests for the catalogue validation function."""

    def _make_assessment(self, **overrides):
        """Create a sample assessment dict with defaults."""
        base = {
            "name": "Test Assessment",
            "url": "https://www.shl.com/products/product-catalog/view/test/",
            "remote_support": "Yes",
            "adaptive_irt": "No",
            "test_type_codes": ["K"],
            "test_types": ["Knowledge & Skills"],
            "test_type": "Knowledge & Skills",
            "description": "A test assessment for testing.",
            "duration": 30,
        }
        base.update(overrides)
        return base

    def test_perfect_catalogue(self):
        assessments = [self._make_assessment(name=f"Test {i}", url=f"https://shl.com/{i}") for i in range(5)]
        report = validate_catalogue(assessments)
        assert report["total"] == 5
        assert report["with_description"] == 5
        assert report["with_duration"] == 5
        assert report["unique_urls"] == 5
        assert len(report["broken_urls"]) == 0

    def test_missing_descriptions(self):
        assessments = [
            self._make_assessment(description="Has description"),
            self._make_assessment(description=""),
            self._make_assessment(description=None),
        ]
        report = validate_catalogue(assessments)
        assert report["with_description"] == 1
        assert len(report["missing_descriptions"]) == 2

    def test_missing_durations(self):
        assessments = [
            self._make_assessment(duration=30),
            self._make_assessment(duration=None),
        ]
        report = validate_catalogue(assessments)
        assert report["with_duration"] == 1

    def test_test_type_distribution(self):
        assessments = [
            self._make_assessment(test_types=["Knowledge & Skills"]),
            self._make_assessment(test_types=["Knowledge & Skills"]),
            self._make_assessment(test_types=["Personality & Behavior"]),
        ]
        report = validate_catalogue(assessments)
        assert report["test_type_distribution"]["Knowledge & Skills"] == 2
        assert report["test_type_distribution"]["Personality & Behavior"] == 1


# --- Tests for Save/Load ---

class TestSaveLoadCatalogue:
    """Tests for catalogue persistence."""

    def test_save_and_load(self, tmp_path):
        assessments = [
            {"name": "Test 1", "url": "https://example.com/1", "duration": 30},
            {"name": "Test 2", "url": "https://example.com/2", "duration": 45},
        ]
        filepath = tmp_path / "test_catalogue.json"

        save_catalogue(assessments, filepath)
        loaded = load_catalogue(filepath)

        assert len(loaded) == 2
        assert loaded[0]["name"] == "Test 1"
        assert loaded[1]["duration"] == 45

    def test_load_nonexistent(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_catalogue(tmp_path / "nonexistent.json")

    def test_save_creates_directory(self, tmp_path):
        filepath = tmp_path / "subdir" / "catalogue.json"
        save_catalogue([{"name": "test"}], filepath)
        assert filepath.exists()


# --- Integration-style Tests for Listing Page Parsing ---

class TestScrapeListingPage:
    """Tests for parsing HTML listing pages (using mock responses)."""

    SAMPLE_LISTING_HTML = """
    <html><body>
    <table>
    <thead><tr>
        <th>Individual Test Solutions</th>
        <th>Remote Testing</th>
        <th>Adaptive/IRT</th>
        <th>Test Type</th>
    </tr></thead>
    <tbody>
    <tr>
        <td class="custom__table-heading__title">
            <a href="/products/product-catalog/view/python-new/">Python (New)</a>
        </td>
        <td class="custom__table-heading__general">
            <span class="catalogue__circle -yes"></span>
        </td>
        <td class="custom__table-heading__general">
            <span class="catalogue__circle"></span>
        </td>
        <td class="custom__table-heading__general product-catalogue__keys">
            <span class="product-catalogue__key">K</span>
        </td>
    </tr>
    <tr>
        <td class="custom__table-heading__title">
            <a href="/products/product-catalog/view/verify-g-plus/">Verify G+</a>
        </td>
        <td class="custom__table-heading__general">
            <span class="catalogue__circle -yes"></span>
        </td>
        <td class="custom__table-heading__general">
            <span class="catalogue__circle -yes"></span>
        </td>
        <td class="custom__table-heading__general product-catalogue__keys">
            <span class="product-catalogue__key">A</span>
        </td>
    </tr>
    </tbody>
    </table>
    </body></html>
    """

    @patch("data.scraper._get_soup")
    def test_parse_listing(self, mock_get_soup):
        from bs4 import BeautifulSoup
        mock_get_soup.return_value = BeautifulSoup(self.SAMPLE_LISTING_HTML, "lxml")

        results = scrape_listing_page("https://example.com", MagicMock())

        assert len(results) == 2

        # First assessment: Python (New)
        python = results[0]
        assert python["name"] == "Python (New)"
        assert "python-new" in python["url"]
        assert python["remote_support"] == "Yes"
        assert python["adaptive_irt"] == "No"
        assert "Knowledge & Skills" in python["test_types"]

        # Second assessment: Verify G+
        verify = results[1]
        assert verify["name"] == "Verify G+"
        assert verify["remote_support"] == "Yes"
        assert verify["adaptive_irt"] == "Yes"
        assert "Ability & Aptitude" in verify["test_types"]

    @patch("data.scraper._get_soup")
    def test_empty_page(self, mock_get_soup):
        from bs4 import BeautifulSoup
        mock_get_soup.return_value = BeautifulSoup("<html><body></body></html>", "lxml")

        results = scrape_listing_page("https://example.com", MagicMock())
        assert results == []

    @patch("data.scraper._get_soup")
    def test_failed_fetch(self, mock_get_soup):
        mock_get_soup.return_value = None
        results = scrape_listing_page("https://example.com", MagicMock())
        assert results == []
