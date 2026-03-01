"""
Generate predictions CSV for the SHL assignment test set.

Output format (exactly as required):
  Column 1: Query (the full query text, repeated for each recommendation)
  Column 2: Assessment_url (one URL per row)

Usage:
  python -m eval.generate_predictions
"""

import csv
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_FILE = Path(__file__).parent.parent / "predictions.csv"


def load_test_data(filepath: Path = None) -> list:
    """Load test queries."""
    if filepath is None:
        filepath = DATA_DIR / "test.json"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Test data not found at {filepath}. "
            "Download from the assignment link and save to data/test.json"
        )
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_predictions(output_path: Path = None):
    """Generate predictions CSV for all test queries."""
    from engine.recommender import recommend

    if output_path is None:
        output_path = OUTPUT_FILE

    test_data = load_test_data()

    rows = []
    for i, entry in enumerate(test_data):
        # Handle different possible test data formats
        if isinstance(entry, dict):
            query = entry.get("query", entry.get("Query", ""))
        elif isinstance(entry, str):
            query = entry
        else:
            continue

        if not query:
            continue

        logger.info(f"Processing test query {i+1}/{len(test_data)}: '{query[:80]}...'")

        results = recommend(query, top_k=10)
        for r in results:
            rows.append({
                "Query": query,
                "Assessment_url": r["url"],
            })

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Query", "Assessment_url"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Saved {len(rows)} predictions to {output_path}")
    print(f"\nGenerated {len(rows)} predictions across {len(test_data)} queries")
    print(f"   Saved to: {output_path}")


if __name__ == "__main__":
    try:
        generate_predictions()
    except FileNotFoundError as e:
        print(f"\n⚠️  {e}")
        print("Please download test.json from the assignment and save to data/test.json")
