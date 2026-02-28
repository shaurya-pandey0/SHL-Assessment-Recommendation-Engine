"""
SHL Product Catalogue Scraper
Scrapes ONLY 'Individual Test Solutions' from:
https://www.shl.com/solutions/products/product-catalog/

The SHL catalogue is JavaScript-rendered, so we use Selenium for listing pages
and requests for detail pages (which render fine server-side).

Target: 377+ assessments with fields:
  - name, url, remote_support, adaptive_support
  - description, duration, test_type (array of full names)
"""

import json
import re
import time
import logging
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.shl.com"
CATALOGUE_URL = f"{BASE_URL}/solutions/products/product-catalog/"
SCRAPER_DIR = Path(__file__).parent

# Test type code → full name mapping
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
}


def scrape_listing_pages_selenium(max_pages: int = 35) -> list[dict]:
    """
    Scrape all listing pages for Individual Test Solutions (type=1)
    using Selenium to handle JS rendering.
    """
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument(f"user-agent={HEADERS['User-Agent']}")

    driver = webdriver.Chrome(options=options)
    all_assessments = []
    seen_urls = set()

    try:
        for page in range(max_pages):
            start = page * 12
            url = f"{CATALOGUE_URL}?start={start}&type=1"
            logger.info(f"Scraping listing page {page + 1} (start={start})...")

            driver.get(url)

            # Wait for the table to load
            try:
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr"))
                )
            except Exception:
                logger.info(f"No table found on page {page + 1}, stopping.")
                break

            # Extract data via JavaScript for reliability
            js_result = driver.execute_script("""
                const results = [];
                const tables = document.querySelectorAll('table');
                // We want the Individual Test Solutions table (type=1)
                // On type=1 pages, there's usually one table
                tables.forEach(table => {
                    const rows = table.querySelectorAll('tbody tr');
                    rows.forEach(row => {
                        const titleLink = row.querySelector('td.custom__table-heading__title a');
                        if (!titleLink) return;
                        const name = titleLink.textContent.trim();
                        const href = titleLink.getAttribute('href');
                        const url = href.startsWith('http') ? href : 'https://www.shl.com' + href;
                        
                        const generalCells = row.querySelectorAll('td.custom__table-heading__general');
                        let remoteSupport = 'No';
                        let adaptiveSupport = 'No';
                        
                        if (generalCells.length >= 1 && generalCells[0].querySelector('span.catalogue__circle.-yes')) {
                            remoteSupport = 'Yes';
                        }
                        if (generalCells.length >= 2 && generalCells[1].querySelector('span.catalogue__circle.-yes')) {
                            adaptiveSupport = 'Yes';
                        }
                        
                        const keySpans = row.querySelectorAll('span.product-catalogue__key');
                        const testTypeCodes = Array.from(keySpans).map(s => s.textContent.trim());
                        
                        results.push({ name, url, remoteSupport, adaptiveSupport, testTypeCodes });
                    });
                });
                return JSON.stringify(results);
            """)

            items = json.loads(js_result)
            if not items:
                logger.info(f"Empty page {page + 1}, stopping.")
                break

            new_count = 0
            for item in items:
                if item["url"] not in seen_urls:
                    seen_urls.add(item["url"])
                    test_type_codes = item["testTypeCodes"]
                    test_types = [TEST_TYPE_MAP.get(c, c) for c in test_type_codes if c]

                    all_assessments.append({
                        "name": item["name"],
                        "url": item["url"],
                        "remote_support": item["remoteSupport"],
                        "adaptive_support": item["adaptiveSupport"],
                        "test_type": test_types,  # ARRAY of full names
                        "description": "",  # filled from detail page
                        "duration": None,   # filled from detail page
                    })
                    new_count += 1

            logger.info(f"  Found {new_count} new assessments (total: {len(all_assessments)})")

            if new_count == 0:
                logger.info("No new assessments found, stopping.")
                break

            time.sleep(1)  # Polite delay

    finally:
        driver.quit()

    logger.info(f"Total assessments from listings: {len(all_assessments)}")
    return all_assessments


def scrape_detail_page(url: str, session: requests.Session) -> dict:
    """
    Scrape a detail page for description and duration.
    Detail pages render server-side so requests works fine.
    """
    try:
        resp = session.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch detail page {url}: {e}")
        return {"description": "", "duration": None}

    soup = BeautifulSoup(resp.text, "lxml")
    result = {"description": "", "duration": None}

    # Try og:description meta tag first — most reliable
    og_desc = soup.find("meta", property="og:description")
    if og_desc and og_desc.get("content"):
        desc = og_desc["content"].strip()
        # Remove the "{Name}: " prefix SHL adds
        colon_idx = desc.find(": ")
        if colon_idx != -1 and colon_idx < 80:
            desc = desc[colon_idx + 2:]
        result["description"] = desc

    # If og:description didn't work, try parsing from page content
    if not result["description"]:
        desc_header = soup.find(string=re.compile(r"Description", re.I))
        if desc_header:
            parent = desc_header.find_parent()
            if parent:
                parts = []
                for sib in parent.find_next_siblings():
                    if sib.name and sib.name.startswith("h"):
                        break
                    text = sib.get_text(strip=True)
                    if text:
                        parts.append(text)
                result["description"] = " ".join(parts)

    # Extract duration
    page_text = soup.get_text()
    duration_match = re.search(
        r"Approximate Completion Time in minutes\s*=\s*(\d+)",
        page_text,
        re.IGNORECASE
    )
    if duration_match:
        result["duration"] = int(duration_match.group(1))
    else:
        # Fallback: look for any "XX minutes" pattern near "completion" or "duration"
        alt_match = re.search(r"(\d+)\s*minutes?", page_text, re.IGNORECASE)
        if alt_match:
            val = int(alt_match.group(1))
            if 1 <= val <= 300:  # Sanity check
                result["duration"] = val

    return result


def scrape_all_detail_pages(assessments: list[dict], delay: float = 0.3) -> list[dict]:
    """Enrich assessments with description and duration from detail pages."""
    session = requests.Session()
    total = len(assessments)
    logger.info(f"Scraping {total} detail pages...")

    for i, assessment in enumerate(assessments):
        if (i + 1) % 25 == 0 or i == 0:
            logger.info(f"  Detail page {i + 1}/{total}...")

        detail = scrape_detail_page(assessment["url"], session)
        assessment["description"] = detail["description"]
        assessment["duration"] = detail["duration"]

        time.sleep(delay)

    return assessments


def validate_catalogue(assessments: list[dict]) -> dict:
    """Validate the scraped data and print a report."""
    report = {
        "total": len(assessments),
        "with_description": sum(1 for a in assessments if a.get("description")),
        "with_duration": sum(1 for a in assessments if a.get("duration") is not None),
        "test_type_distribution": {},
        "missing_descriptions": [],
        "missing_durations": [],
    }

    for a in assessments:
        if not a.get("description"):
            report["missing_descriptions"].append(a["name"])
        if a.get("duration") is None:
            report["missing_durations"].append(a["name"])

        for tt in a.get("test_type", []):
            report["test_type_distribution"][tt] = report["test_type_distribution"].get(tt, 0) + 1

    return report


def print_report(report: dict):
    """Print a formatted validation report."""
    total = report["total"]
    print("\n" + "=" * 60)
    print("SHL CATALOGUE SCRAPE REPORT")
    print("=" * 60)
    print(f"Total assessments:      {total}")
    print(f"With descriptions:      {report['with_description']} ({report['with_description']/max(total,1)*100:.0f}%)")
    print(f"With durations:         {report['with_duration']} ({report['with_duration']/max(total,1)*100:.0f}%)")
    print(f"Missing descriptions:   {len(report['missing_descriptions'])}")
    print(f"Missing durations:      {len(report['missing_durations'])}")
    print(f"\nTest Type Distribution:")
    for tt, count in sorted(report["test_type_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {tt}: {count}")
    if total >= 377:
        print(f"\n✅ PASS: {total} >= 377 assessments scraped")
    else:
        print(f"\n❌ FAIL: Only {total} assessments (need >= 377)")
    print("=" * 60 + "\n")


def save_catalogue(assessments: list[dict], filepath: Path = None) -> Path:
    """Save catalogue to JSON."""
    if filepath is None:
        filepath = SCRAPER_DIR / "catalogue.json"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(assessments, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(assessments)} assessments to {filepath}")
    return filepath


def load_catalogue(filepath: Path = None) -> list[dict]:
    """Load catalogue from JSON."""
    if filepath is None:
        filepath = SCRAPER_DIR / "catalogue.json"
    if not filepath.exists():
        raise FileNotFoundError(f"Catalogue not found at {filepath}. Run the scraper first.")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    logger.info("Starting SHL catalogue scrape (Individual Test Solutions only)...")

    # Step 1: Scrape listings with Selenium (type=1 only)
    assessments = scrape_listing_pages_selenium()

    # Step 2: Enrich with detail page data
    assessments = scrape_all_detail_pages(assessments)

    # Step 3: Save
    save_catalogue(assessments)

    # Step 4: Validate
    report = validate_catalogue(assessments)
    print_report(report)
