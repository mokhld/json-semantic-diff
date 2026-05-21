"""Unit tests for ``compare_batch`` and ``compare_batch_pairs`` (F5).

The batch helpers exist to amortise the per-call comparator setup and warm
the embedding cache across multiple comparisons.  These tests cover the
correctness surface: empty input, identical-input batches, order
preservation, and equivalence with single-call ``compare()``.
"""

from __future__ import annotations

from typing import Any

import pytest

from json_semantic_diff import (
    ComparisonResult,
    STEDConfig,
    compare,
    compare_batch,
    compare_batch_pairs,
)

# ---------------------------------------------------------------------------
# compare_batch
# ---------------------------------------------------------------------------


class TestCompareBatch:
    def test_empty_lefts_returns_empty_list(self) -> None:
        """An empty ``lefts`` list returns an empty result list."""
        result = compare_batch([], {"a": 1})
        assert result == []

    def test_three_identical_inputs_all_score_one(self) -> None:
        """Three identical lefts against the same right all score 1.0."""
        payload: dict[str, Any] = {"user_name": "Alice", "age": 30}
        results = compare_batch([payload, payload, payload], payload)

        assert len(results) == 3
        for r in results:
            assert isinstance(r, ComparisonResult)
            assert r.similarity_score == pytest.approx(1.0)

    def test_order_preserved(self) -> None:
        """Results are returned in the same order as the input lefts."""
        right: dict[str, Any] = {"target": "value"}
        lefts: list[Any] = [
            {"target": "value"},  # identical → score 1.0
            {"completely_different_key": "x"},  # low score
            {"target": "value"},  # identical → score 1.0
        ]
        results = compare_batch(lefts, right)

        assert len(results) == 3
        assert results[0].similarity_score == pytest.approx(1.0)
        assert results[2].similarity_score == pytest.approx(1.0)
        # Middle entry should be strictly lower than the identical ones.
        assert results[1].similarity_score < results[0].similarity_score

    def test_matches_single_compare(self) -> None:
        """Batch results equal what we would get from individual compare() calls."""
        right: dict[str, Any] = {"user_name": "Alice"}
        lefts: list[Any] = [
            {"user_name": "Alice"},
            {"userName": "Alice"},
            {"address": "123 Main St"},
        ]

        batch = compare_batch(lefts, right)
        singles = [compare(left, right) for left in lefts]

        assert len(batch) == len(singles)
        for b, s in zip(batch, singles, strict=True):
            # Cache reuse must not change the score.
            assert b.similarity_score == pytest.approx(s.similarity_score)
            assert b.key_mappings == s.key_mappings
            assert b.unmatched_left == s.unmatched_left
            assert b.unmatched_right == s.unmatched_right

    def test_config_is_respected(self) -> None:
        """A passed STEDConfig flows through to every comparison in the batch."""
        config = STEDConfig(null_equals_missing=True)
        lefts: list[Any] = [{"a": 1, "b": None}, {"a": 1}]
        right: dict[str, Any] = {"a": 1}
        results = compare_batch(lefts, right, config=config)
        # With null_equals_missing=True, {"a":1,"b":None} ≡ {"a":1}
        for r in results:
            assert r.similarity_score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compare_batch_pairs
# ---------------------------------------------------------------------------


class TestCompareBatchPairs:
    def test_empty_pairs_returns_empty_list(self) -> None:
        """An empty pairs list returns an empty result list."""
        result = compare_batch_pairs([])
        assert result == []

    def test_three_identical_pairs_score_one(self) -> None:
        """Three identical (left, right) pairs all score 1.0."""
        payload: dict[str, Any] = {"user_name": "Alice"}
        pairs: list[tuple[Any, Any]] = [(payload, payload)] * 3
        results = compare_batch_pairs(pairs)

        assert len(results) == 3
        for r in results:
            assert r.similarity_score == pytest.approx(1.0)

    def test_order_preserved(self) -> None:
        """Pair results are returned in input order."""
        pairs: list[tuple[Any, Any]] = [
            ({"a": 1}, {"a": 1}),
            ({"a": 1}, {"completely_unrelated": 2}),
            ({"a": 1}, {"a": 1}),
        ]
        results = compare_batch_pairs(pairs)

        assert len(results) == 3
        assert results[0].similarity_score == pytest.approx(1.0)
        assert results[2].similarity_score == pytest.approx(1.0)
        assert results[1].similarity_score < results[0].similarity_score

    def test_matches_single_compare(self) -> None:
        """Batched pair results equal individual compare() calls."""
        pairs: list[tuple[Any, Any]] = [
            ({"a": 1}, {"a": 1}),
            ({"user_name": "x"}, {"userName": "x"}),
            ({"x": [1, 2, 3]}, {"x": [1, 2, 3]}),
        ]
        batch = compare_batch_pairs(pairs)
        singles = [compare(left, right) for left, right in pairs]
        assert len(batch) == len(singles)
        for b, s in zip(batch, singles, strict=True):
            assert b.similarity_score == pytest.approx(s.similarity_score)

    def test_config_is_respected(self) -> None:
        """A passed STEDConfig flows through to every pair in the batch."""
        config = STEDConfig(null_equals_missing=True)
        pairs: list[tuple[Any, Any]] = [
            ({"a": 1, "b": None}, {"a": 1}),
            ({"a": 2, "z": None}, {"a": 2}),
        ]
        for r in compare_batch_pairs(pairs, config=config):
            assert r.similarity_score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Re-export wiring
# ---------------------------------------------------------------------------


class TestReExports:
    def test_top_level_imports(self) -> None:
        """compare_batch and compare_batch_pairs are importable from the package."""
        import json_semantic_diff as jsd

        assert hasattr(jsd, "compare_batch")
        assert hasattr(jsd, "compare_batch_pairs")
        assert "compare_batch" in jsd.__all__
        assert "compare_batch_pairs" in jsd.__all__
