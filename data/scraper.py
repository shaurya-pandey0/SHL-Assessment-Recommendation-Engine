"""
SHL Product Catalogue Scraper

Scrapes all assessments from SHL's product catalogue at:
https://www.shl.com/solutions/products/product-catalog/

Two types of assessments:
  - type=2: Pre-packaged Job Solutions (~12 pages)
  - type=1: Individual Test Solutions (~32 pages)

Each listing page shows 12 items with:
  - Name & URL
  - Remote Testing support (green circle = yes)
  - Adaptive/IRT support (green circle = yes)
  - Test type codes (A, B, C, D, E, K, P, S)

Each detail page adds:
  - Full description
  - Duration (minutes)
  - Job levels
  - Languages
"""

import json
import re
import time
import logging
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://www.shl.com"
CATALOGUE_URL = f"{BASE_URL}/solutions/products/product-catalog/"
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

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def _get_soup(url: str, session: requests.Session, retries: int = 3) -> Optional[BeautifulSoup]:
    """Fetch a URL and return a BeautifulSoup object."""
    for attempt in range(retries):
        try:
            resp = session.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "lxml")
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt + 1}/{retries} failed for {url}: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def _parse_duration(text: str) -> Optional[int]:
    """Extract duration in minutes from assessment length text."""
    if not text:
        return None
    match = re.search(r"(\d+)", text)
    return int(match.group(1)) if match else None


def _parse_test_type_codes(codes: list[str]) -> list[str]:
    """Convert test type letter codes to full names."""
    return [TEST_TYPE_MAP.get(code.strip(), code.strip()) for code in codes if code.strip()]


def scrape_listing_page(url: str, session: requests.Session) -> list[dict]:
    """Scrape a single listing page and return basic assessment info."""
    soup = _get_soup(url, session)
    if not soup:
        logger.error(f"Failed to fetch listing page: {url}")
        return []

    assessments = []
    tables = soup.select("table")

    for table in tables:
        rows = table.select("tbody tr")
        for row in rows:
            # Get name and URL
            title_cell = row.select_one("td.custom__table-heading__title a")
            if not title_cell:
                continue

            name = title_cell.get_text(strip=True)
            href = title_cell.get("href", "")
            detail_url = href if href.startswith("http") else f"{BASE_URL}{href}"

            # Get Remote Testing and Adaptive/IRT from the general columns
            general_cells = row.select("td.custom__table-heading__general")

            remote_support = "No"
            adaptive_irt = "No"

            if len(general_cells) >= 1:
                # First general cell: Remote Testing
                if general_cells[0].select_one("span.catalogue__circle.-yes"):
                    remote_support = "Yes"

            if len(general_cells) >= 2:
                # Second general cell: Adaptive/IRT
                if general_cells[1].select_one("span.catalogue__circle.-yes"):
                    adaptive_irt = "Yes"

            # Get test type codes
            test_type_cell = row.select_one("td.product-catalogue__keys")
            if not test_type_cell:
                # Fallback: try the last general cell
                if len(general_cells) >= 3:
                    test_type_cell = general_cells[-1]

            test_type_codes = []
            if test_type_cell:
                key_spans = test_type_cell.select("span.product-catalogue__key")
                test_type_codes = [span.get_text(strip=True) for span in key_spans]

            test_types = _parse_test_type_codes(test_type_codes)

            assessments.append({
                "name": name,
                "url": detail_url,
                "remote_support": remote_support,
                "adaptive_irt": adaptive_irt,
                "test_type_codes": test_type_codes,
                "test_types": test_types,
                "test_type": ", ".join(test_types) if test_types else "Unknown",
            })

    return assessments


def scrape_detail_page(url: str, session: requests.Session) -> dict:
    """Scrape a detail page for description and duration."""
    soup = _get_soup(url, session)
    if not soup:
        return {"description": "", "duration": None, "job_levels": "", "languages": ""}

    result = {
        "description": "",
        "duration": None,
        "job_levels": "",
        "languages": "",
    }

    # Find the main content area
    content = soup.select_one("div.product-catalogue-training, main, article, .content")
    if not content:
        content = soup

    # Extract description
    desc_header = content.find(string=re.compile(r"Description", re.I))
    if desc_header:
        parent = desc_header.find_parent()
        if parent:
            # Get all following siblings until next header
            description_parts = []
            for sibling in parent.find_next_siblings():
                if sibling.name and sibling.name.startswith("h"):
                    break
                text = sibling.get_text(strip=True)
                if text:
                    description_parts.append(text)
            result["description"] = " ".join(description_parts)

    # If description is still empty, try og:description meta tag
    if not result["description"]:
        og_desc = soup.find("meta", property="og:description")
        if og_desc:
            desc_text = og_desc.get("content", "")
            # Remove the assessment name prefix if present
            result["description"] = desc_text

    # Extract assessment length / duration
    length_header = content.find(string=re.compile(r"Assessment length|Completion Time", re.I))
    if length_header:
        parent = length_header.find_parent()
        if parent:
            # Look in text of parent and siblings
            text_block = parent.get_text() if parent else ""
            for sibling in parent.find_next_siblings():
                if sibling.name and sibling.name.startswith("h"):
                    break
                text_block += " " + sibling.get_text()
            result["duration"] = _parse_duration(text_block)

    # Extract job levels
    level_header = content.find(string=re.compile(r"Job levels", re.I))
    if level_header:
        parent = level_header.find_parent()
        if parent:
            for sibling in parent.find_next_siblings():
                if sibling.name and sibling.name.startswith("h"):
                    break
                text = sibling.get_text(strip=True)
                if text:
                    result["job_levels"] = text
                    break

    # Extract languages
    lang_header = content.find(string=re.compile(r"Languages", re.I))
    if lang_header:
        parent = lang_header.find_parent()
        if parent:
            for sibling in parent.find_next_siblings():
                if sibling.name and sibling.name.startswith("h"):
                    break
                text = sibling.get_text(strip=True)
                if text:
                    result["languages"] = text
                    break

    return result


def scrape_all_assessments(
    types: list[int] = None,
    max_pages_per_type: int = 50,
    delay: float = 1.0,
) -> list[dict]:
    """
    Scrape all assessments from the SHL product catalogue.

    Args:
        types: List of assessment types to scrape (1=Individual, 2=Pre-packaged).
               Defaults to both [1, 2].
        max_pages_per_type: Maximum number of pages to scrape per type.
        delay: Delay between requests in seconds.

    Returns:
        List of assessment dictionaries.
    """
    if types is None:
        types = [1, 2]

    session = requests.Session()
    all_assessments = []
    seen_urls = set()

    for assessment_type in types:
        type_name = "Individual Test Solutions" if assessment_type == 1 else "Pre-packaged Job Solutions"
        logger.info(f"Scraping {type_name} (type={assessment_type})...")

        page = 0
        consecutive_empty = 0

        while page < max_pages_per_type:
            start = page * 12
            url = f"{CATALOGUE_URL}?start={start}&type={assessment_type}"

            logger.info(f"  Page {page + 1} (start={start})...")
            assessments = scrape_listing_page(url, session)

            if not assessments:
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    logger.info(f"  No more results. Stopping at page {page + 1}.")
                    break
                page += 1
                time.sleep(delay)
                continue

            consecutive_empty = 0

            for assessment in assessments:
                if assessment["url"] not in seen_urls:
                    seen_urls.add(assessment["url"])
                    assessment["catalogue_type"] = type_name
                    all_assessments.append(assessment)

            page += 1
            time.sleep(delay)

        logger.info(f"  Found {len([a for a in all_assessments if a['catalogue_type'] == type_name])} {type_name}")

    logger.info(f"\nTotal unique assessments from listings: {len(all_assessments)}")

    # Now scrape detail pages for descriptions and durations
    logger.info(f"\nScraping detail pages for {len(all_assessments)} assessments...")
    for i, assessment in enumerate(all_assessments):
        if (i + 1) % 20 == 0 or i == 0:
            logger.info(f"  Detail page {i + 1}/{len(all_assessments)}...")

        detail = scrape_detail_page(assessment["url"], session)
        assessment.update(detail)
        time.sleep(delay * 0.5)  # Slightly faster for detail pages

    return all_assessments


def save_catalogue(assessments: list[dict], filepath: Path = None) -> Path:
    """Save assessments to a JSON file."""
    if filepath is None:
        filepath = DATA_DIR / "catalogue.json"

    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(assessments, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(assessments)} assessments to {filepath}")
    return filepath


def load_catalogue(filepath: Path = None) -> list[dict]:
    """Load assessments from a JSON file."""
    if filepath is None:
        filepath = DATA_DIR / "catalogue.json"

    if not filepath.exists():
        raise FileNotFoundError(
            f"Catalogue not found at {filepath}. Run the scraper first."
        )

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_catalogue(assessments: list[dict]) -> dict:
    """Validate the scraped catalogue data and return a quality report."""
    report = {
        "total": len(assessments),
        "with_description": sum(1 for a in assessments if a.get("description")),
        "with_duration": sum(1 for a in assessments if a.get("duration") is not None),
        "with_test_types": sum(1 for a in assessments if a.get("test_types")),
        "with_remote_info": sum(1 for a in assessments if a.get("remote_support") in ("Yes", "No")),
        "with_adaptive_info": sum(1 for a in assessments if a.get("adaptive_irt") in ("Yes", "No")),
        "unique_urls": len(set(a.get("url", "") for a in assessments)),
        "broken_urls": [],
        "missing_descriptions": [],
        "test_type_distribution": {},
    }

    for a in assessments:
        if not a.get("url") or not a["url"].startswith("http"):
            report["broken_urls"].append(a.get("name", "unknown"))
        if not a.get("description"):
            report["missing_descriptions"].append(a.get("name", "unknown"))

        # Count test type distribution
        for tt in a.get("test_types", []):
            report["test_type_distribution"][tt] = report["test_type_distribution"].get(tt, 0) + 1

    return report


def print_validation_report(report: dict):
    """Print a formatted validation report."""
    print("\n" + "=" * 60)
    print("SHL CATALOGUE VALIDATION REPORT")
    print("=" * 60)
    print(f"Total assessments:       {report['total']}")
    print(f"Unique URLs:             {report['unique_urls']}")
    print(f"With descriptions:       {report['with_description']} ({report['with_description']/max(report['total'],1)*100:.0f}%)")
    print(f"With durations:          {report['with_duration']} ({report['with_duration']/max(report['total'],1)*100:.0f}%)")
    print(f"With test types:         {report['with_test_types']} ({report['with_test_types']/max(report['total'],1)*100:.0f}%)")
    print(f"With remote info:        {report['with_remote_info']} ({report['with_remote_info']/max(report['total'],1)*100:.0f}%)")
    print(f"Broken URLs:             {len(report['broken_urls'])}")
    print(f"Missing descriptions:    {len(report['missing_descriptions'])}")
    print(f"\nTest Type Distribution:")
    for tt, count in sorted(report["test_type_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {tt}: {count}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    logger.info("Starting SHL catalogue scrape...")
    assessments = scrape_all_assessments(delay=0.8)
    save_catalogue(assessments)
    report = validate_catalogue(assessments)
    print_validation_report(report)
