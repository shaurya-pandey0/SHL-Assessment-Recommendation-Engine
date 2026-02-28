"""
Browser-based scraper for SHL catalogue.

Since the SHL page is JavaScript-rendered, this script collects
the data via browser automation. The actual browser interaction
is done by the agent using browser subagent tools.

This module provides utilities for collecting and merging data
scraped from the browser into the catalogue.json format.
"""

import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent

# Test type code to full name mapping
TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}

# JavaScript to extract data from a catalogue page
EXTRACT_JS = """
(() => {
  const results = [];
  const tables = document.querySelectorAll('table');
  tables.forEach(table => {
    const rows = table.querySelectorAll('tbody tr');
    rows.forEach(row => {
      const titleLink = row.querySelector('td.custom__table-heading__title a');
      if (!titleLink) return;
      const name = titleLink.textContent.trim();
      const url = titleLink.href;
      const generalCells = row.querySelectorAll('td.custom__table-heading__general');
      let remoteSupport = 'No';
      let adaptiveIRT = 'No';
      if (generalCells.length >= 1 && generalCells[0].querySelector('span.catalogue__circle.-yes')) {
        remoteSupport = 'Yes';
      }
      if (generalCells.length >= 2 && generalCells[1].querySelector('span.catalogue__circle.-yes')) {
        adaptiveIRT = 'Yes';
      }
      const keySpans = row.querySelectorAll('span.product-catalogue__key');
      const testTypeCodes = Array.from(keySpans).map(s => s.textContent.trim());
      results.push({ name, url, remoteSupport, adaptiveIRT, testTypeCodes });
    });
  });
  return JSON.stringify(results);
})()
"""

# JavaScript to extract detail page info
DETAIL_JS = """
(() => {
  const getText = (selector) => {
    const el = document.querySelector(selector);
    return el ? el.textContent.trim() : '';
  };
  
  // Get all section content
  const sections = {};
  const headers = document.querySelectorAll('h4, h3, .product-catalogue__heading');
  headers.forEach(h => {
    const title = h.textContent.trim().toLowerCase();
    let content = '';
    let sibling = h.nextElementSibling;
    while (sibling && !['H3', 'H4'].includes(sibling.tagName)) {
      content += sibling.textContent.trim() + ' ';
      sibling = sibling.nextElementSibling;
    }
    sections[title] = content.trim();
  });
  
  // Extract duration from "Approximate Completion Time in minutes = XX"
  let duration = null;
  const lengthText = sections['assessment length'] || '';
  const durationMatch = lengthText.match(/(\\d+)/);
  if (durationMatch) duration = parseInt(durationMatch[1]);
  
  return JSON.stringify({
    description: sections['description'] || '',
    duration: duration,
    job_levels: sections['job levels'] || '',
    languages: sections['languages'] || '',
  });
})()
"""


def process_browser_data(raw_items: list[dict]) -> list[dict]:
    """
    Process raw browser-extracted items into catalogue format.
    
    Args:
        raw_items: List of dicts from browser JavaScript extraction.
        
    Returns:
        List of catalogue-formatted assessment dicts.
    """
    assessments = []
    
    for item in raw_items:
        test_type_codes = item.get("testTypeCodes", [])
        test_types = [TEST_TYPE_MAP.get(c, c) for c in test_type_codes if c]
        
        assessments.append({
            "name": item["name"],
            "url": item["url"],
            "remote_support": item.get("remoteSupport", "No"),
            "adaptive_irt": item.get("adaptiveIRT", "No"),
            "test_type_codes": test_type_codes,
            "test_types": test_types,
            "test_type": ", ".join(test_types) if test_types else "Unknown",
            "description": item.get("description", ""),
            "duration": item.get("duration"),
        })
    
    return assessments


def merge_catalogue(existing: list[dict], new_items: list[dict]) -> list[dict]:
    """Merge new items into existing catalogue, avoiding duplicates."""
    seen_urls = {a["url"] for a in existing}
    
    for item in new_items:
        if item["url"] not in seen_urls:
            seen_urls.add(item["url"])
            existing.append(item)
    
    return existing


def save_catalogue(assessments: list[dict], filepath: Path = None) -> Path:
    """Save catalogue to JSON file."""
    if filepath is None:
        filepath = DATA_DIR / "catalogue.json"
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(assessments, f, indent=2, ensure_ascii=False)
    
    return filepath


def load_catalogue(filepath: Path = None) -> list[dict]:
    """Load catalogue from JSON file."""
    if filepath is None:
        filepath = DATA_DIR / "catalogue.json"
    
    if not filepath.exists():
        return []
    
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def get_pagination_urls() -> list[str]:
    """Generate all pagination URLs for both types."""
    base = "https://www.shl.com/solutions/products/product-catalog/"
    urls = []
    
    # Type 2: Pre-packaged Job Solutions (~12 pages)
    for page in range(12):
        urls.append(f"{base}?start={page * 12}&type=2")
    
    # Type 1: Individual Test Solutions (~32 pages)
    for page in range(35):
        urls.append(f"{base}?start={page * 12}&type=1")
    
    return urls
