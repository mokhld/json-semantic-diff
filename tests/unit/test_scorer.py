"""Unit tests for ConsistencyScorer and consistency_score() API function.

Tests are organized in TDD order:
1. ConsistencyScorer.compute() behavior (edge cases, formula, config forwarding)
2. consistency_score() public API wrapper
"""

from __future__ import annotations

import pytest

from json_semantic_diff.algorithm.config import STEDConfig
from json_semantic_diff.scorer import ConsistencyScorer


class TestConsistencyScorerEdgeCases:
    """Edge case behavior for empty and single-document inputs."""

    def test_empty_list_returns_1(self) -> None:
        """Empty list is trivially consistent — returns 1.0."""
        scorer = ConsistencyScorer()
        assert scorer.compute([]) == pytest.approx(1.0)

    def test_single_doc_returns_1(self) -> None:
        """Single document is trivially consistent — returns 1.0."""
        scorer = ConsistencyScorer()
        assert scorer.compute([{"a": 1}]) == pytest.approx(1.0)

    def test_two_identical_docs_returns_1(self) -> None:
        """Two identical documents score 1.0."""
        scorer = ConsistencyScorer()
        assert scorer.compute([{"k": "v"}, {"k": "v"}]) == pytest.approx(1.0)


class TestConsistencyScorerBounds:
    """Property: score is always in [0.0, 1.0] for any list of JSON docs."""

    @pytest.mark.parametrize(
        "docs",
        [
            [{"a": 1}, {"a": 1}],
            [{"a": 1}, {"b": 2}, {"c": 3}],
            [{}, {}, {}],
            [1, 2, 3],
            ["hello", "world", "foo"],
            [None, None, None],
            [{"x": [1, 2, 3]}, {"x": [3, 2, 1]}],
        ],
    )
    def test_score_is_in_unit_interval(self, docs: list) -> None:
        scorer = ConsistencyScorer()
        score = scorer.compute(docs)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestConsistencyScorerFormula:
    """Verify max(0, mean - std) formula behavior."""

    def test_identical_docs_score_1(self) -> None:
        """Three identical documents score exactly 1.0."""
        scorer = ConsistencyScorer()
        doc = {"a": 1}
        result = scorer.compute([doc, doc, doc])
        assert result == pytest.approx(1.0)

    def test_structurally_different_score_low(self) -> None:
        """Structurally different documents score well below 0.5."""
        scorer = ConsistencyScorer()
        docs = [
            {"name": "Alice"},
            {"product": "Widget"},
            {"city": "Paris"},
        ]
        result = scorer.compute(docs)
        assert result < 0.5

    def test_score_always_in_range(self) -> None:
        """Score for diverse documents is always in [0.0, 1.0]."""
        scorer = ConsistencyScorer()
        docs = [{"a": 1}, {"b": 2}, {"c": 3}]
        result = scorer.compute(docs)
        assert 0.0 <= result <= 1.0

    def test_erratic_scores_lower_than_consistent(self) -> None:
        """Erratic generator scores lower than consistent generator."""
        scorer = ConsistencyScorer()
        consistent_docs = [{"a": 1}, {"a": 1}, {"a": 1}]
        erratic_docs = [{"a": 1}, {"x": 99}, {"a": 1}]
        consistent_score = scorer.compute(consistent_docs)
        erratic_score = scorer.compute(erratic_docs)
        assert consistent_score > erratic_score


class TestConsistencyScorerComparatorReuse:
    """Verify comparator is reused (not recreated) across pairwise calls."""

    def test_comparator_reused_across_pairs(self) -> None:
        """ConsistencyScorer reuses a single comparator; compare called N*(N-1)/2 times.

        For N=3 docs, there are C(3,2) = 3 pairs.
        """
        scorer = ConsistencyScorer()
        docs = [{"a": 1}, {"b": 2}, {"c": 3}]

        call_count = 0
        original_compare = scorer._comparator.compare

        def spy_compare(left, right):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            return original_compare(left, right)

        scorer._comparator.compare = spy_compare  # type: ignore[method-assign]
        scorer.compute(docs)

        assert call_count == 3, f"Expected 3 pairwise calls for N=3, got {call_count}"


class TestConsistencyScorerConfigForwarding:
    """Verify config and max_cache_size are forwarded to the internal comparator."""

    def test_config_forwarded(self) -> None:
        """STEDConfig is forwarded to the internal comparator."""
        config = STEDConfig(type_coercion=True)
        scorer = ConsistencyScorer(config=config)
        assert scorer._comparator._config.type_coercion is True

    def test_max_cache_size_forwarded(self) -> None:
        """max_cache_size is forwarded to the internal EmbeddingCache."""
        scorer = ConsistencyScorer(max_cache_size=64)
        # EmbeddingCache is stored as _comparator._backend
        assert scorer._comparator._backend.max_size == 64


class TestConsistencyScoreAPIFunction:
    """Tests for the module-level consistency_score() API function in api.py."""

    def test_api_consistency_score_identical(self) -> None:
        """consistency_score() returns 1.0 for identical documents."""
        from json_semantic_diff.api import consistency_score

        result = consistency_score([{"a": 1}, {"a": 1}])
        assert result == pytest.approx(1.0)

    def test_api_consistency_score_different(self) -> None:
        """consistency_score() returns < 0.5 for structurally different documents."""
        from json_semantic_diff.api import consistency_score

        result = consistency_score([{"a": 1}, {"z": 99}])
        assert result < 0.5

    def test_api_consistency_score_with_config(self) -> None:
        """consistency_score() accepts config parameter without error."""
        from json_semantic_diff.api import consistency_score

        config = STEDConfig()
        # Should not raise any exception
        result = consistency_score([{"a": 1}, {"a": 1}], config=config)
        assert isinstance(result, float)
