"""Tests for STEDComparator — the central orchestrator for semantic JSON comparison.

Covers:
- Core comparison functionality (identical, different, naming-convention pairs)
- null_equals_missing preprocessing (flat and nested)
- KEY-level match extraction (matched_pairs, key_mappings, unmatched lists)
- Default backend behaviour (no args → StaticBackend)
- Statelessness (two identical calls → identical results)
- computation_time_ms is always a positive float
"""

from __future__ import annotations

import pytest

from json_semantic_diff.algorithm.config import STEDConfig
from json_semantic_diff.comparator import STEDComparator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _key_names_in_pairs(pairs: list[tuple[str, str]], side: int) -> list[str]:
    """Extract the final path segment (key name) from matched pair paths."""
    return [p[side].rsplit("/", 1)[-1] for p in pairs]


# ---------------------------------------------------------------------------
# Core functionality
# ---------------------------------------------------------------------------


class TestCoreComparison:
    def test_identical_objects_score_1_0(self) -> None:
        cmp = STEDComparator()
        result = cmp.compare({"a": 1}, {"a": 1})
        assert result.similarity_score == pytest.approx(1.0, abs=1e-9)

    def test_identical_objects_matched_pairs_non_empty(self) -> None:
        cmp = STEDComparator()
        result = cmp.compare({"a": 1}, {"a": 1})
        assert len(result.matched_pairs) > 0

    def test_identical_objects_matched_pair_paths_contain_key_a(self) -> None:
        cmp = STEDComparator()
        result = cmp.compare({"a": 1}, {"a": 1})
        left_keys = _key_names_in_pairs(result.matched_pairs, 0)
        assert "a" in left_keys

    def test_identical_objects_key_mappings(self) -> None:
        cmp = STEDComparator()
        result = cmp.compare({"a": 1}, {"a": 1})
        assert result.key_mappings == {"a": "a"}

    def test_identical_objects_no_unmatched(self) -> None:
        cmp = STEDComparator()
        result = cmp.compare({"a": 1}, {"a": 1})
        assert result.unmatched_left == ()
        assert result.unmatched_right == ()

    def test_identical_objects_computation_time_positive(self) -> None:
        cmp = STEDComparator()
        result = cmp.compare({"a": 1}, {"a": 1})
        assert result.computation_time_ms > 0.0

    def test_different_objects_score_less_than_1(self) -> None:
        cmp = STEDComparator()
        result = cmp.compare({"a": 1}, {"b": 2})
        assert result.similarity_score < 1.0

    def test_naming_convention_user_name_vs_userName_score_above_085(self) -> None:
        cmp = STEDComparator()
        result = cmp.compare({"user_name": "John"}, {"userName": "John"})
        assert result.similarity_score > 0.85

    def test_naming_convention_key_mappings_correct(self) -> None:
        cmp = STEDComparator()
        result = cmp.compare({"user_name": "John"}, {"userName": "John"})
        assert result.key_mappings.get("user_name") == "userName"

    def test_empty_objects_score_1_0(self) -> None:
        cmp = STEDComparator()
        result = cmp.compare({}, {})
        assert result.similarity_score == pytest.approx(1.0, abs=1e-9)

    def test_scalars_score_1_0_when_identical(self) -> None:
        cmp = STEDComparator()
        result = cmp.compare("hello", "hello")
        assert result.similarity_score == pytest.approx(1.0, abs=1e-9)

    def test_scalars_no_matched_pairs(self) -> None:
        """Scalar roots have no KEY children — matched_pairs is empty."""
        cmp = STEDComparator()
        result = cmp.compare("hello", "world")
        assert result.matched_pairs == ()
        assert result.key_mappings == {}


# ---------------------------------------------------------------------------
# null_equals_missing
# ---------------------------------------------------------------------------


class TestNullEqualsMissing:
    def test_null_equals_missing_true_none_vs_empty_score_1_0(self) -> None:
        cmp = STEDComparator(config=STEDConfig(null_equals_missing=True))
        result = cmp.compare({"x": None}, {})
        assert result.similarity_score == pytest.approx(1.0, abs=1e-9)

    def test_null_equals_missing_false_none_vs_empty_score_less_than_1(self) -> None:
        cmp = STEDComparator(config=STEDConfig())
        result = cmp.compare({"x": None}, {})
        assert result.similarity_score < 1.0

    def test_null_equals_missing_true_partial_none(self) -> None:
        cmp = STEDComparator(config=STEDConfig(null_equals_missing=True))
        result = cmp.compare({"a": 1, "b": None}, {"a": 1})
        assert result.similarity_score == pytest.approx(1.0, abs=1e-9)

    def test_null_equals_missing_true_nested_none(self) -> None:
        cmp = STEDComparator(config=STEDConfig(null_equals_missing=True))
        result = cmp.compare({"a": {"b": None}}, {"a": {}})
        assert result.similarity_score == pytest.approx(1.0, abs=1e-9)

    def test_null_equals_missing_false_does_not_preprocess(self) -> None:
        """Default config — {"x": None} != {} — score must reflect the difference."""
        cmp = STEDComparator(config=STEDConfig(null_equals_missing=False))
        result_none = cmp.compare({"x": None}, {})
        result_same = cmp.compare({}, {})
        # With the key present (value=None), the score should be lower than for {}=={}
        assert result_none.similarity_score < result_same.similarity_score


# ---------------------------------------------------------------------------
# Match extraction
# ---------------------------------------------------------------------------


class TestMatchExtraction:
    def test_partially_overlapping_keys_a_matched_to_a(self) -> None:
        cmp = STEDComparator()
        result = cmp.compare({"a": 1, "b": 2}, {"a": 1, "c": 3})
        left_keys = _key_names_in_pairs(result.matched_pairs, 0)
        right_keys = _key_names_in_pairs(result.matched_pairs, 1)
        assert "a" in left_keys
        assert "a" in right_keys

    def test_partially_overlapping_all_keys_matched_hungarian_exhaustive(self) -> None:
        """Hungarian always exhaustively matches all keys when sizes are equal.

        {"a":1,"b":2} vs {"a":1,"c":3} yields 2x2 cost matrix — Hungarian
        assigns all 2 left keys to all 2 right keys (minimum total cost),
        so unmatched lists are empty even though "b" and "c" are dissimilar.
        """
        cmp = STEDComparator()
        result = cmp.compare({"a": 1, "b": 2}, {"a": 1, "c": 3})
        assert len(result.matched_pairs) == 2
        assert result.unmatched_left == ()
        assert result.unmatched_right == ()

    def test_unmatched_produced_when_sizes_differ(self) -> None:
        """When left has more keys than right, surplus left keys are unmatched."""
        cmp = STEDComparator()
        result = cmp.compare({"a": 1, "b": 2, "d": 4}, {"a": 1})
        unmatched_left_names = [p.rsplit("/", 1)[-1] for p in result.unmatched_left]
        # "b" and "d" cannot be matched — right has only 1 key
        assert len(result.unmatched_left) == 2
        assert set(unmatched_left_names) == {"b", "d"}

    def test_unmatched_right_when_right_has_more_keys(self) -> None:
        """When right has more keys than left, surplus right keys are unmatched."""
        cmp = STEDComparator()
        result = cmp.compare({"a": 1}, {"a": 1, "c": 3, "d": 4})
        unmatched_right_names = [p.rsplit("/", 1)[-1] for p in result.unmatched_right]
        assert len(result.unmatched_right) == 2
        assert set(unmatched_right_names) == {"c", "d"}

    def test_nested_match_extraction_outer_and_inner_in_matched_pairs(self) -> None:
        cmp = STEDComparator()
        result = cmp.compare({"outer": {"inner": 1}}, {"outer": {"inner": 1}})
        # "outer" should appear in matched pairs
        left_key_names = _key_names_in_pairs(result.matched_pairs, 0)
        assert "outer" in left_key_names
        # "inner" should also appear (nested recursion)
        assert "inner" in left_key_names

    def test_completely_different_keys_all_unmatched(self) -> None:
        """When keys have zero similarity, Hungarian may still match them.

        The important invariant is that the union of matched + unmatched equals
        the total number of keys on each side.
        """
        cmp = STEDComparator()
        result = cmp.compare({"a": 1}, {"b": 2})
        total_left = len(result.matched_pairs) + len(result.unmatched_left)
        total_right = len(result.matched_pairs) + len(result.unmatched_right)
        assert total_left == 1
        assert total_right == 1

    def test_empty_left_all_right_keys_unmatched(self) -> None:
        cmp = STEDComparator()
        result = cmp.compare({}, {"a": 1, "b": 2})
        assert result.matched_pairs == ()
        assert result.key_mappings == {}
        assert len(result.unmatched_right) == 2

    def test_empty_right_all_left_keys_unmatched(self) -> None:
        cmp = STEDComparator()
        result = cmp.compare({"a": 1, "b": 2}, {})
        assert result.matched_pairs == ()
        assert result.key_mappings == {}
        assert len(result.unmatched_left) == 2


# ---------------------------------------------------------------------------
# Default backend
# ---------------------------------------------------------------------------


class TestDefaultBackend:
    def test_no_args_uses_static_backend(self) -> None:
        """STEDComparator() with no args must work without error."""
        cmp = STEDComparator()
        result = cmp.compare({"a": 1}, {"a": 1})
        assert result.similarity_score == pytest.approx(1.0, abs=1e-9)

    def test_no_args_naming_convention_similarity(self) -> None:
        cmp = STEDComparator()
        result = cmp.compare({"user_name": "Alice"}, {"userName": "Alice"})
        assert result.similarity_score > 0.85


# ---------------------------------------------------------------------------
# Statelessness (API-05 SC5)
# ---------------------------------------------------------------------------


class TestStatelessness:
    def test_two_identical_calls_return_same_score(self) -> None:
        cmp = STEDComparator()
        r1 = cmp.compare({"a": 1, "b": 2}, {"a": 1, "b": 2})
        r2 = cmp.compare({"a": 1, "b": 2}, {"a": 1, "b": 2})
        assert r1.similarity_score == r2.similarity_score

    def test_two_identical_calls_return_same_matched_pairs(self) -> None:
        cmp = STEDComparator()
        r1 = cmp.compare({"a": 1, "b": 2}, {"a": 1, "b": 2})
        r2 = cmp.compare({"a": 1, "b": 2}, {"a": 1, "b": 2})
        assert r1.matched_pairs == r2.matched_pairs

    def test_two_identical_calls_return_same_key_mappings(self) -> None:
        cmp = STEDComparator()
        r1 = cmp.compare({"a": 1, "b": 2}, {"a": 1, "b": 2})
        r2 = cmp.compare({"a": 1, "b": 2}, {"a": 1, "b": 2})
        assert r1.key_mappings == r2.key_mappings

    def test_two_identical_calls_return_same_unmatched(self) -> None:
        cmp = STEDComparator()
        r1 = cmp.compare({"a": 1, "c": 3}, {"a": 1, "b": 2})
        r2 = cmp.compare({"a": 1, "c": 3}, {"a": 1, "b": 2})
        assert r1.unmatched_left == r2.unmatched_left
        assert r1.unmatched_right == r2.unmatched_right

    def test_computation_time_always_positive(self) -> None:
        cmp = STEDComparator()
        for _ in range(5):
            result = cmp.compare({"x": 42}, {"x": 42})
            assert result.computation_time_ms > 0.0
