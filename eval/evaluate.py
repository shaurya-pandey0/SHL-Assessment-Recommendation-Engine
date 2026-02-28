"""
Evaluation framework for the SHL Assessment Recommendation Engine.

Computes:
  - Recall@K per query
  - Mean Recall@K across all queries
  - MAP@K (Mean Average Precision)

Usage:
  python -m eval.evaluate          # Evaluate on train data
  python -m eval.evaluate --verbose # Show per-query details
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


def recall_at_k(recommended_urls: list[str], relevant_urls: list[str], k: int = 10) -> float:
    """What fraction of relevant assessments appear in top K recommendations?"""
    recommended_set = set(recommended_urls[:k])
    relevant_set = set(relevant_urls)
    if not relevant_set:
        return 0.0
    return len(recommended_set & relevant_set) / len(relevant_set)


def average_precision_at_k(recommended_urls: list[str], relevant_urls: list[str], k: int = 10) -> float:
    """Compute Average Precision at K for a single query."""
    relevant_set = set(relevant_urls)
    if not relevant_set:
        return 0.0

    hits = 0
    sum_precisions = 0.0

    for i, url in enumerate(recommended_urls[:k]):
        if url in relevant_set:
            hits += 1
            precision_at_i = hits / (i + 1)
            sum_precisions += precision_at_i

    if not relevant_set:
        return 0.0
    return sum_precisions / min(len(relevant_set), k)


def mean_recall_at_k(results: list[tuple], k: int = 10) -> float:
    """Average Recall@K across all queries."""
    recalls = [recall_at_k(rec, rel, k) for rec, rel in results]
    if not recalls:
        return 0.0
    return sum(recalls) / len(recalls)


def mean_average_precision(results: list[tuple], k: int = 10) -> float:
    """MAP@K across all queries."""
    aps = [average_precision_at_k(rec, rel, k) for rec, rel in results]
    if not aps:
        return 0.0
    return sum(aps) / len(aps)


def load_train_data(filepath: Path = None) -> list[dict]:
    """Load labelled train data."""
    if filepath is None:
        filepath = DATA_DIR / "train.json"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Train data not found at {filepath}. "
            "Download from the assignment link and save to data/train.json"
        )
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_url(url: str) -> str:
    """Normalize URL for comparison (remove trailing slash, lowercase)."""
    url = url.strip().rstrip("/").lower()
    # Handle both URL patterns
    url = url.replace("solutions/products/product-catalog/", "products/product-catalog/")
    return url


def evaluate_on_train(verbose: bool = False) -> dict:
    """
    Run evaluation on the train dataset.

    Returns dict with metrics and per-query breakdown.
    """
    from engine.recommender import recommend

    train_data = load_train_data()
    results = []
    per_query = []

    for i, entry in enumerate(train_data):
        # Handle different possible train data formats
        if isinstance(entry, dict):
            query = entry.get("query", entry.get("Query", ""))
            relevant_urls = entry.get("relevant_urls", entry.get("Expected", []))
            if isinstance(relevant_urls, str):
                relevant_urls = [relevant_urls]
        else:
            continue

        if not query:
            continue

        # Get recommendations
        recs = recommend(query, top_k=10)
        rec_urls = [r["url"] for r in recs]

        # Normalize URLs for fair comparison
        rec_urls_norm = [normalize_url(u) for u in rec_urls]
        relevant_urls_norm = [normalize_url(u) for u in relevant_urls]

        r_at_10 = recall_at_k(rec_urls_norm, relevant_urls_norm, k=10)
        ap_at_10 = average_precision_at_k(rec_urls_norm, relevant_urls_norm, k=10)

        results.append((rec_urls_norm, relevant_urls_norm))

        query_result = {
            "query_idx": i + 1,
            "query": query[:100],
            "recall@10": r_at_10,
            "ap@10": ap_at_10,
            "n_relevant": len(relevant_urls),
            "n_hits": len(set(rec_urls_norm) & set(relevant_urls_norm)),
        }
        per_query.append(query_result)

        if verbose:
            print(f"\n  Query {i+1}: '{query[:80]}...'")
            print(f"  Recall@10: {r_at_10:.3f} | AP@10: {ap_at_10:.3f}")
            print(f"  Relevant: {len(relevant_urls)} | Hits: {query_result['n_hits']}")
            if r_at_10 < 1.0:
                # Show what was missed
                missed = set(relevant_urls_norm) - set(rec_urls_norm)
                if missed:
                    print(f"  Missed URLs: {list(missed)[:3]}")

    # Compute aggregate metrics
    mean_r = mean_recall_at_k(results, k=10)
    map_10 = mean_average_precision(results, k=10)

    report = {
        "mean_recall@10": mean_r,
        "MAP@10": map_10,
        "n_queries": len(results),
        "per_query": per_query,
    }

    return report


def print_report(report: dict):
    """Print formatted evaluation report."""
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"Queries evaluated:   {report['n_queries']}")
    print(f"Mean Recall@10:      {report['mean_recall@10']:.4f}")
    print(f"MAP@10:              {report['MAP@10']:.4f}")

    print(f"\nPer-query breakdown:")
    print(f"  {'#':<3} {'Recall@10':<12} {'AP@10':<10} {'Hits':<6} {'Query':<50}")
    print(f"  {'-'*80}")
    for q in report["per_query"]:
        print(
            f"  {q['query_idx']:<3} {q['recall@10']:<12.3f} "
            f"{q['ap@10']:<10.3f} {q['n_hits']:<6} {q['query'][:50]}"
        )
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SHL recommendation engine")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    args = parser.parse_args()

    try:
        report = evaluate_on_train(verbose=args.verbose)
        print_report(report)
    except FileNotFoundError as e:
        print(f"\n⚠️  {e}")
        print("Please download train.json from the assignment and save to data/train.json")
