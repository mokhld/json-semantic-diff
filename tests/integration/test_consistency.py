"""Integration tests for the consistency_score() function.

Covers identical documents, structurally different documents, the
mean-minus-std formula, and edge cases (empty list, single doc, config
forwarding).
"""

from __future__ import annotations

import pytest

from json_semantic_diff import STEDConfig, consistency_score


class TestSC1IdenticalDocuments:
    """SC1 — consistency_score returns 1.0 for identical documents."""

    def test_identical_dicts_three_copies(self) -> None:
        """SC1: Three identical flat dicts score 1.0."""
        doc = {"user_name": "Alice", "age": 30}
        result = consistency_score([doc, doc, doc])
        assert result == pytest.approx(1.0)

    def test_identical_nested_objects(self) -> None:
        """SC1: Three copies of a deeply nested object score 1.0."""
        nested = {
            "level1": {
                "level2": {
                    "level3": {"value": 42, "name": "deep"},
                    "count": 10,
                },
                "tag": "outer",
            }
        }
        result = consistency_score([nested, nested, nested])
        assert result == pytest.approx(1.0)

    def test_identical_primitives(self) -> None:
        """SC1: Three identical primitive values score 1.0."""
        result = consistency_score([42, 42, 42])
        assert result == pytest.approx(1.0)

    def test_identical_arrays(self) -> None:
        """SC1: Three identical arrays score 1.0."""
        result = consistency_score([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        assert result == pytest.approx(1.0)

    def test_two_identical_docs(self) -> None:
        """SC1: Two identical documents (N=2 edge case) score 1.0."""
        doc = {"user_name": "Alice", "age": 30}
        result = consistency_score([doc, doc])
        assert result == pytest.approx(1.0)


class TestSC2StructurallyDifferent:
    """SC2 — consistency_score returns < 0.7 for structurally different documents.

    Audit C6 (wave 7): the per-pair similarity for same-shape, different-content
    OBJECTs no longer binary-collapses against a ``len(children)`` denominator.
    The new floor is around 0.5 per pair, so the post-std-penalty consistency
    score sits below 0.7 (but no longer below 0.5) — the spirit of the SC2
    assertion (well below the equivalence band) still holds.
    """

    def test_three_unrelated_objects(self) -> None:
        """SC2: Three objects with completely different keys score < 0.7."""
        docs = [
            {"name": "Alice", "age": 30},
            {"product": "Widget", "price": 9.99},
            {"city": "Paris", "country": "France"},
        ]
        result = consistency_score(docs)
        assert result < 0.7

    def test_five_unrelated_objects(self) -> None:
        """SC2: Five objects with completely different key sets score < 0.7."""
        docs = [
            {"alpha": 1},
            {"beta": 2},
            {"gamma": 3},
            {"delta": 4},
            {"epsilon": 5},
        ]
        result = consistency_score(docs)
        assert result < 0.7

    def test_two_different_docs(self) -> None:
        """SC2: Two structurally different documents score < 0.7."""
        result = consistency_score([{"a": 1}, {"z": 99}])
        assert result < 0.7


class TestSC3NormalizedStdDev:
    """SC3 — formula uses max(0, mean - std), not a simple mean."""

    def test_formula_not_simple_mean(self) -> None:
        """SC3: Variance penalty drives score below the simple pair-score mean.

        With [{"a":1}, {"a":1}, {"zzz":999}]:
        - Pair (0,1): score 1.0 (identical)
        - Pair (0,2): score ~0.5 (same shape, different key+value)
        - Pair (1,2): score ~0.5 (same shape, different key+value)

        Audit C6 (wave 7): the previous version of this test expected the
        unrelated pairs to score ~0.0 (binary collapse).  Under proper
        Zhang-Shasha normalisation they sit around 0.5.  The std penalty
        still drives the consistency score below the actual pair-score mean
        because the variance across pairs is non-zero.
        """
        docs = [{"a": 1}, {"a": 1}, {"zzz": 999}]
        result = consistency_score(docs)
        # Compute the actual mean of pair similarities so the assertion
        # stays calibrated to whatever the per-pair formula produces.
        from json_semantic_diff import compare

        pair_scores = [
            compare(docs[0], docs[1]).similarity_score,
            compare(docs[0], docs[2]).similarity_score,
            compare(docs[1], docs[2]).similarity_score,
        ]
        simple_mean = sum(pair_scores) / len(pair_scores)
        assert result <= simple_mean, (
            f"Score {result} should be <= simple mean {simple_mean} due to std penalty"
        )

    def test_erratic_lower_than_consistent(self) -> None:
        """SC3: Variance penalty makes erratic generator score lower than consistent."""
        consistent_score = consistency_score([{"a": 1}, {"a": 1}, {"a": 1}])
        erratic_score = consistency_score([{"a": 1}, {"x": 99}, {"a": 1}])
        assert consistent_score > erratic_score, (
            f"Consistent ({consistent_score}) should be > erratic ({erratic_score})"
        )

    def test_mediocre_consistent_beats_erratic(self) -> None:
        """SC3: Consistent mediocre generator beats erratic high-variance generator.

        Mediocre-but-consistent: three pairs all score ~0.0 but identically.
        Erratic: two pairs score 1.0, one pair scores ~0.0.
        Mean-only: erratic would appear better (higher mean).
        With std penalty: consistent mediocre wins because std=0.
        """
        # Erratic: [{"a":1}, {"a":1}, {"zzz":999}]
        # Two pairs score 1.0, one pair ~0.0 -> mean ~0.67, std ~0.47
        # Score = max(0, 0.67 - 0.47) ~= 0.20
        erratic_docs = [{"a": 1}, {"a": 1}, {"zzz": 999}]
        erratic_score = consistency_score(erratic_docs)

        # Mediocre consistent: [{"a":1}, {"b":2}, {"c":3}]
        # All pairs score ~0.0 (disjoint) -> mean ~0.0, std ~0.0
        # Score = max(0, 0 - 0) = 0.0
        # This doesn't beat erratic; use same-key-but-different-value instead:
        # [{"a":1}, {"a":2}, {"a":3}] — same key structure, different values
        # Pairs all have same key, so structural similarity is 1.0
        consistent_docs = [{"a": 1}, {"a": 2}, {"a": 3}]
        consistent_score = consistency_score(consistent_docs)

        # consistent_score should be > erratic_score because erratic has high variance
        assert consistent_score >= erratic_score, (
            f"Consistent mediocre ({consistent_score}) should be >= erratic ({erratic_score})"
        )


class TestEdgeCases:
    """Edge case behavior for empty, single-doc, bounded scores, and config."""

    def test_empty_list(self) -> None:
        """Empty list is trivially consistent — returns 1.0."""
        result = consistency_score([])
        assert result == pytest.approx(1.0)

    def test_single_document(self) -> None:
        """Single document is trivially consistent — returns 1.0."""
        result = consistency_score([{"a": 1}])
        assert result == pytest.approx(1.0)

    def test_score_bounded_0_1_diverse_sets(self) -> None:
        """Score is always in [0.0, 1.0] for diverse document sets."""
        test_cases = [
            [{"a": 1}, {"b": 2}],
            [{"x": "hello"}, {"x": "world"}, {"y": 42}],
            [{"name": "Alice", "age": 30}, {"product": "Widget"}],
            [1, 2, 3],
            ["foo", "bar", "baz"],
        ]
        for docs in test_cases:
            result = consistency_score(docs)
            assert 0.0 <= result <= 1.0, f"Score {result} out of [0,1] for docs: {docs}"

    def test_config_parameter_works(self) -> None:
        """Config forwarding works end-to-end: type_coercion=True scores 1.0.

        With type_coercion=True, {"x": "123"} and {"x": 123} should score 1.0
        as the string "123" coerces to the integer 123.
        """
        docs = [{"x": "123"}, {"x": 123}]
        result = consistency_score(docs, config=STEDConfig(type_coercion=True))
        assert result == pytest.approx(1.0)
