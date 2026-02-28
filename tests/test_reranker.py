"""
Tests for the re-ranker and test-type balancer.

Tests:
  - Duration hard filter
  - Test type score boost
  - Balance mode (technical + behavioral interleaving)
  - Combined filters
"""

import pytest
from engine.reranker import rerank, balance_test_types


def _make_candidate(name="Test", duration=30, remote="Yes",
                    test_type=None, score=0.5, url="https://shl.com/test"):
    return {
        "name": name,
        "url": url,
        "duration": duration,
        "remote_support": remote,
        "test_type": test_type or [],
        "score": score,
    }


class TestRerank:
    def test_duration_filter(self):
        candidates = [
            _make_candidate("Short", duration=20, score=0.8),
            _make_candidate("Long", duration=60, score=0.9, url="https://shl.com/long"),
            _make_candidate("Medium", duration=30, score=0.7, url="https://shl.com/med"),
        ]
        parsed = {"max_duration": 30, "test_types_needed": [], "requires_balance": False}
        results = rerank(candidates, parsed, top_k=10)
        names = [r["name"] for r in results]
        assert "Long" not in names
        assert "Short" in names
        assert "Medium" in names

    def test_null_duration_kept(self):
        candidates = [
            _make_candidate("NoDur", duration=None, score=0.8),
            _make_candidate("HasDur", duration=60, score=0.9, url="https://shl.com/h"),
        ]
        parsed = {"max_duration": 30, "test_types_needed": [], "requires_balance": False}
        results = rerank(candidates, parsed, top_k=10)
        names = [r["name"] for r in results]
        assert "NoDur" in names
        assert "HasDur" not in names

    def test_type_boost(self):
        candidates = [
            _make_candidate("Wrong", test_type=["Simulations"], score=0.8, url="https://a"),
            _make_candidate("Right", test_type=["Knowledge & Skills"], score=0.7, url="https://b"),
        ]
        parsed = {"max_duration": None, "test_types_needed": ["Knowledge & Skills"], "requires_balance": False}
        results = rerank(candidates, parsed, top_k=10)
        # Right (0.7 + 0.15) = 0.85 > Wrong (0.8)
        assert results[0]["name"] == "Right"

    def test_no_constraints(self):
        candidates = [
            _make_candidate(score=0.9, url="https://a"),
            _make_candidate(score=0.8, url="https://b"),
        ]
        parsed = {"max_duration": None, "test_types_needed": [], "requires_balance": False}
        results = rerank(candidates, parsed, top_k=10)
        assert len(results) == 2


class TestBalanceTestTypes:
    def test_interleave(self):
        tech = [
            _make_candidate("T1", test_type=["Knowledge & Skills"], score=0.9, url="https://t1"),
            _make_candidate("T2", test_type=["Knowledge & Skills"], score=0.8, url="https://t2"),
            _make_candidate("T3", test_type=["Knowledge & Skills"], score=0.7, url="https://t3"),
        ]
        behav = [
            _make_candidate("B1", test_type=["Personality & Behavior"], score=0.85, url="https://b1"),
            _make_candidate("B2", test_type=["Personality & Behavior"], score=0.75, url="https://b2"),
            _make_candidate("B3", test_type=["Personality & Behavior"], score=0.65, url="https://b3"),
        ]
        results = balance_test_types(tech + behav, top_k=6)
        names = [r["name"] for r in results]
        # Should have mix of T and B
        tech_count = sum(1 for n in names if n.startswith("T"))
        behav_count = sum(1 for n in names if n.startswith("B"))
        assert tech_count >= 2
        assert behav_count >= 2

    def test_only_technical(self):
        tech = [
            _make_candidate("T1", test_type=["Knowledge & Skills"], score=0.9, url="https://t1"),
            _make_candidate("T2", test_type=["Knowledge & Skills"], score=0.8, url="https://t2"),
        ]
        results = balance_test_types(tech, top_k=5)
        assert len(results) == 2
        assert all(r["name"].startswith("T") for r in results)

    def test_only_behavioral(self):
        behav = [
            _make_candidate("B1", test_type=["Personality & Behavior"], score=0.9, url="https://b1"),
        ]
        results = balance_test_types(behav, top_k=5)
        assert len(results) == 1


class TestRerankWithBalance:
    def test_balance_mode_activated(self):
        candidates = [
            _make_candidate("T1", test_type=["Knowledge & Skills"], score=0.95, url="https://t1"),
            _make_candidate("T2", test_type=["Knowledge & Skills"], score=0.90, url="https://t2"),
            _make_candidate("T3", test_type=["Knowledge & Skills"], score=0.85, url="https://t3"),
            _make_candidate("T4", test_type=["Knowledge & Skills"], score=0.80, url="https://t4"),
            _make_candidate("T5", test_type=["Knowledge & Skills"], score=0.75, url="https://t5"),
            _make_candidate("B1", test_type=["Personality & Behavior"], score=0.50, url="https://b1"),
            _make_candidate("B2", test_type=["Personality & Behavior"], score=0.45, url="https://b2"),
            _make_candidate("B3", test_type=["Personality & Behavior"], score=0.40, url="https://b3"),
        ]
        parsed = {
            "max_duration": None,
            "test_types_needed": ["Knowledge & Skills", "Personality & Behavior"],
            "requires_balance": True,
        }
        results = rerank(candidates, parsed, top_k=6)

        # With balance: should have behavioral items even though they have lower scores
        names = [r["name"] for r in results]
        behav_in_results = sum(1 for n in names if n.startswith("B"))
        assert behav_in_results >= 2, f"Expected >= 2 behavioral, got {behav_in_results}: {names}"
