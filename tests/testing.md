# Testing Guide

This document explains how to run the tests for the SHL Assessment Recommendation Engine.

## Why running tests directly fails
If you try to run a test file directly like this:
```bash
python tests/test_api.py
```
You will get a `ModuleNotFoundError: No module named 'engine'`. This happens because Python doesn't know where the root of your project is, so it can't find your `engine`, `api`, or `scraper` folders.

## The Solution: Use pytest as a module

Always use `python -m pytest` from the root directory (`c:\Users\PC\Desktop\Project\SHL Assignment`). This sets the Python path correctly so all imports work.

### Run all tests
```bash
python -m pytest tests/ -v
```

### Run a single test file
```bash
python -m pytest tests/test_api.py -v
python -m pytest tests/test_query_parser.py -v
python -m pytest tests/test_reranker.py -v
```

### Run a single test class
```bash
python -m pytest tests/test_api.py::TestHealthEndpoint -v
```

### Run a single test function
```bash
python -m pytest tests/test_api.py::TestRecommendEndpoint::test_recommend_returns_200 -v
```

## Useful Flags

By default, pytest **captures all print output** — you only see PASSED/FAILED. Use these flags to control behavior:

| Flag | What it does |
|------|-------------|
| `-v` | Verbose — show full test names |
| `-s` | **Show print() output** (disables capture) |
| `--tb=long` | Full tracebacks on failure |
| `--tb=short` | Compact tracebacks (default in this project) |
| `-x` | Stop on first failure |
| `-k "keyword"` | Run only tests matching a keyword |

### Example: see all output
```bash
python -m pytest tests/test_api.py -v -s
```

### Example: run only tests with "python" in the name
```bash
python -m pytest tests/test_recommender.py -v -s -k "python"
```

## Adding Custom Print Statements

If an interviewer asks you to add debug output or inspect results, just add `print()` in the test and run with `-s`:

```python
def test_recommend_returns_results(self, client):
    response = client.post("/recommend", json={"query": "Python"})
    data = response.json()

    # Custom debug output — visible with -s flag
    print(f"\nGot {len(data['recommended_assessments'])} results:")
    for r in data["recommended_assessments"]:
        print(f"  → {r['name']} | {r['duration']} min | {r['test_type']}")

    assert len(data["recommended_assessments"]) > 0
```

Then run:
```bash
python -m pytest tests/test_api.py::TestRecommendEndpoint::test_recommend_returns_results -v -s
```

## Changing Parameters On The Fly

To test with different queries or top_k values, you can modify the test directly or add a new test:

```python
def test_custom_query(self, setup_pipeline):
    from engine.recommender import recommend
    results = recommend("Java developer who manages teams", top_k=5)

    print(f"\nCustom query results:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['name']} | {', '.join(r['test_type'])} | {r['duration']} min")

    assert len(results) > 0
    assert len(results) <= 5
```

## Notes
- Tests that touch embeddings (like `test_api.py`, `test_recommender.py`, `test_search.py`) will take a few seconds on their first run. This is because they create real embeddings dynamically from small sample catalogues (3-5 items) during the test setup. This is completely expected and ensures the whole pipeline works end-to-end.
- The `-s` flag is your best friend in interviews — it lets you show exactly what the system is doing.
