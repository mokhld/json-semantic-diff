"""Discrimination benchmark tests for FastEmbedBackend model validation.

These tests validate that all-MiniLM-L6-v2 (the default FastEmbedBackend model
after Phase 8 benchmark) can distinguish semantically related key names
(>0.90) from unrelated ones (<0.90) through the full STEDComparator stack.

**Threshold rationale (Audit C6 / wave 7):**
The STED algorithm with a single-key object and equal values (e.g.
``{"key_a": "x"}`` vs ``{"key_b": "x"}``) has a mathematically bounded
score range -- the minimum possible score is determined by the per-pair
matched cost divided by the Zhang-Shasha denominator
``sum(subtree_size(c) for c in children)`` (= 2 for one KEY + one SCALAR).

Under the old ``len(children)=1`` denominator the floor was 0.5, but the
raw distance binary-collapsed past it: ``1 - min(1, raw/1)`` clipped to
0.5 for any raw >= 0.5.  Under proper Zhang-Shasha normalisation (wave 7)
the floor moves up to 0.75 for equal-value comparisons (``raw = w_s *
(1 - sim)``, max 0.5, divided by denom 2).  The thresholds were
rebalanced accordingly:

- Related keys (naming-convention equivalents): score > 0.90
  Actual: all-MiniLM-L6-v2 scores 1.0 for all camelCase/snake_case pairs.
- Unrelated keys (semantically different): score < 0.90
  Actual: all-MiniLM-L6-v2 scores 0.78-0.86 for the unrelated pairs.
- Discrimination gap: >= 0.10
  Actual: ~0.14 single-pair gap, ~0.17 multi-pair average.

The gap compresses by half under the new normaliser; pre-wave-7 numbers
in PROJECT.md (~0.29 single-pair, ~0.36 multi-pair) were measured against
the binary-collapsed scores and are no longer the relevant baseline.

All tests skip cleanly when fastembed is not installed.
"""

import pytest

pytest.importorskip(
    "fastembed", reason="fastembed extra not installed — skip accuracy tests"
)

from json_semantic_diff.backends.fastembed import FastEmbedBackend
from json_semantic_diff.comparator import STEDComparator


@pytest.fixture(scope="module")
def fastembed_comparator() -> STEDComparator:
    """Single FastEmbedBackend instance shared across all discrimination tests.

    Using module scope avoids repeated ONNX model initialization (~1-2s each).
    The EmbeddingCache inside STEDComparator is also reused, so subsequent
    tests benefit from cached embeddings.
    """
    return STEDComparator(backend=FastEmbedBackend())


class TestRelatedKeysAboveThreshold:
    """SC2 (revised): Semantically equivalent key names must score > 0.90.

    All naming-convention pairs (snake_case vs camelCase) score 1.0 with
    all-MiniLM-L6-v2 through the full STED stack.  The threshold is 0.90
    (vs the original ROADMAP 0.85) to reflect the empirically observed scores.
    """

    @pytest.mark.parametrize(
        ("left_key", "right_key"),
        [
            ("user_name", "userName"),
            ("first_name", "firstName"),
            ("email_address", "emailAddress"),
            ("phone_number", "phoneNumber"),
            ("created_at", "createdAt"),
        ],
    )
    def test_related_keys_score_above_090(
        self,
        fastembed_comparator: STEDComparator,
        left_key: str,
        right_key: str,
    ) -> None:
        result = fastembed_comparator.compare({left_key: "x"}, {right_key: "x"})
        assert result.similarity_score > 0.90, (
            f"Related keys '{left_key}' vs '{right_key}' "
            f"scored {result.similarity_score:.4f} (expected > 0.90)"
        )


class TestUnrelatedKeysBelowThreshold:
    """SC2 (audit C6 / wave 7): Structurally different key names score < 0.90.

    Under the Zhang-Shasha denominator (sum of children subtree sizes), the
    minimum possible score for single-key equal-value objects is 0.75 — the
    matched-pair raw distance ``w_s * (1 - sim)`` (max 0.5) is divided by
    denom 2 (KEY + SCALAR).  The empirical upper bound for semantically
    unrelated keys via all-MiniLM-L6-v2 is ~0.86, so 0.90 stays achievable
    while still capturing "below the equivalence band".
    """

    @pytest.mark.parametrize(
        ("left_key", "right_key"),
        [
            ("user_name", "address"),
            ("email", "age"),
            ("created_at", "price"),
            ("first_name", "total_amount"),
            ("phone_number", "description"),
        ],
    )
    def test_unrelated_keys_score_below_090(
        self,
        fastembed_comparator: STEDComparator,
        left_key: str,
        right_key: str,
    ) -> None:
        result = fastembed_comparator.compare({left_key: "x"}, {right_key: "x"})
        assert result.similarity_score < 0.90, (
            f"Unrelated keys '{left_key}' vs '{right_key}' "
            f"scored {result.similarity_score:.4f} (expected < 0.90)"
        )


class TestDiscriminationGap:
    """SC3 (audit C6 / wave 7): Gap between related and unrelated >= 0.10.

    Under the Zhang-Shasha denominator the addressable score range for
    same-shape single-key OBJECTs is [0.75, 1.0] rather than [0.5, 1.0],
    halving the empirical discrimination gap.  The pre-wave-7 threshold
    (>= 0.25) was calibrated against the binary-collapsed unrelated scores
    and is no longer the relevant baseline.  ``>= 0.10`` keeps the gate
    sensitive to a future regression where the related/unrelated bands
    overlap completely.
    """

    def test_discrimination_gap_gate(
        self, fastembed_comparator: STEDComparator
    ) -> None:
        """If this test fails, the default model must be re-evaluated."""
        related = fastembed_comparator.compare(
            {"user_name": "x"}, {"userName": "x"}
        ).similarity_score
        unrelated = fastembed_comparator.compare(
            {"user_name": "x"}, {"address": "x"}
        ).similarity_score
        gap = related - unrelated
        assert gap >= 0.10, (
            f"Discrimination gap {gap:.4f} < 0.10 threshold -- "
            "all-MiniLM-L6-v2 fails short-phrase key discrimination. "
            "Re-evaluate default model selection."
        )

    def test_gap_across_multiple_pairs(
        self, fastembed_comparator: STEDComparator
    ) -> None:
        """Average gap across multiple pairs must also be >= 0.10."""
        related_pairs = [
            ("user_name", "userName"),
            ("first_name", "firstName"),
            ("email_address", "emailAddress"),
        ]
        unrelated_pairs = [
            ("user_name", "address"),
            ("email", "age"),
            ("created_at", "price"),
        ]
        related_scores = [
            fastembed_comparator.compare({lk: "x"}, {rk: "x"}).similarity_score
            for lk, rk in related_pairs
        ]
        unrelated_scores = [
            fastembed_comparator.compare({lk: "x"}, {rk: "x"}).similarity_score
            for lk, rk in unrelated_pairs
        ]
        avg_related = sum(related_scores) / len(related_scores)
        avg_unrelated = sum(unrelated_scores) / len(unrelated_scores)
        avg_gap = avg_related - avg_unrelated
        assert avg_gap >= 0.10, (
            f"Average discrimination gap {avg_gap:.4f} < 0.10 -- "
            f"related avg: {avg_related:.4f}, unrelated avg: {avg_unrelated:.4f}"
        )
