"""Integration tests for the public API surface.

All imports are from the top-level ``json_semantic_diff`` package — never from
internal submodules.  Covers ComparisonResult richness, is_equivalent(),
similarity_score(), config modes, and statelessness.
"""

from __future__ import annotations

from json_semantic_diff import (
    ComparisonResult,
    STEDConfig,
    compare,
    is_equivalent,
    similarity_score,
)


class TestSC1ComparisonResultRichness:
    """SC1 — compare() returns a fully populated ComparisonResult."""

    def test_compare_identical_returns_populated_result(self) -> None:
        result = compare({"a": 1}, {"a": 1})
        assert isinstance(result, ComparisonResult)
        assert result.matched_pairs is not None
        assert len(result.matched_pairs) > 0
        assert result.key_mappings is not None
        assert len(result.key_mappings) > 0
        assert result.unmatched_left is not None  # may be empty tuple, not None
        assert result.unmatched_right is not None
        assert result.computation_time_ms is not None
        assert result.computation_time_ms > 0

    def test_similarity_score_is_1_for_identical(self) -> None:
        result = compare({"a": 1}, {"a": 1})
        assert result.similarity_score == 1.0

    def test_result_is_frozen(self) -> None:
        """ComparisonResult is a frozen dataclass — mutation must raise."""
        import pytest

        result = compare({"a": 1}, {"a": 1})
        with pytest.raises((AttributeError, TypeError)):
            result.similarity_score = 0.5  # type: ignore[misc]


class TestSC2IsEquivalent:
    """SC2 — is_equivalent() detects semantic equivalence with static backend."""

    def test_semantic_equivalence_detected(self) -> None:
        assert (
            is_equivalent({"user_name": "John"}, {"userName": "John"}, threshold=0.85)
            is True
        )

    def test_returns_bool(self) -> None:
        result = is_equivalent({"a": 1}, {"a": 1})
        assert isinstance(result, bool)

    def test_genuinely_different_not_equivalent(self) -> None:
        assert is_equivalent({"a": 1}, {"b": 2}) is False


class TestSC3SimilarityScore:
    """SC3 — similarity_score() returns a float in [0.0, 1.0]."""

    def test_returns_float_in_range(self) -> None:
        score = similarity_score({"a": 1}, {"b": 2})
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_identical_returns_1(self) -> None:
        score = similarity_score({"x": 42}, {"x": 42})
        assert score == 1.0

    def test_score_never_exceeds_1(self) -> None:
        score = similarity_score({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 2, "c": 3})
        assert score <= 1.0

    def test_score_never_below_0(self) -> None:
        score = similarity_score({"a": 1}, {"z": 99})
        assert score >= 0.0


class TestSC4ConfigModes:
    """SC4 — STEDConfig type_coercion and null_equals_missing work correctly."""

    def test_type_coercion(self) -> None:
        result = compare(
            {"x": "123"}, {"x": 123}, config=STEDConfig(type_coercion=True)
        )
        assert result.similarity_score == 1.0

    def test_null_equals_missing(self) -> None:
        result = compare({"x": None}, {}, config=STEDConfig(null_equals_missing=True))
        assert result.similarity_score == 1.0

    def test_type_coercion_off_by_default(self) -> None:
        """Without coercion, "123" and 123 should NOT score 1.0."""
        result = compare({"x": "123"}, {"x": 123})
        assert result.similarity_score < 1.0

    def test_null_equals_missing_off_by_default(self) -> None:
        """Without the flag, {x: null} and {} should NOT score 1.0."""
        result = compare({"x": None}, {})
        assert result.similarity_score < 1.0


class TestSC5NoGlobalState:
    """SC5 — Repeated calls with the same frozen config return identical results."""

    def test_identical_calls_return_identical_results(self) -> None:
        config = STEDConfig()
        r1 = compare({"a": 1, "b": "hello"}, {"a": 1, "c": "world"}, config=config)
        r2 = compare({"a": 1, "b": "hello"}, {"a": 1, "c": "world"}, config=config)
        assert r1.similarity_score == r2.similarity_score
        assert r1.matched_pairs == r2.matched_pairs
        assert r1.key_mappings == r2.key_mappings
        assert r1.unmatched_left == r2.unmatched_left
        assert r1.unmatched_right == r2.unmatched_right
        # computation_time_ms may differ slightly — do not compare

    def test_fresh_comparator_per_call(self) -> None:
        """Each compare() call must produce consistent results regardless of order."""
        a = {"name": "Alice", "age": 30}
        b = {"name": "Bob", "age": 25}

        score_ab_1 = similarity_score(a, b)
        score_ab_2 = similarity_score(a, b)
        assert score_ab_1 == score_ab_2
