"""Wide-object behaviour (audit gap T7).

Objects with more than 100 keys exercise the Hungarian key-matching path.
These tests guard against future regressions of audit item I2 (Hungarian
sparsity / cubic blow-ups) and ensure the score itself stays sensible at
scale.  They are deliberately *not* benchmarks — they run on every CI
test pass and use a deliberately loose wall-clock ceiling so they only
fire on truly broken cubic regressions, not on small environment-driven
variance.
"""

from __future__ import annotations

import time

from json_semantic_diff import compare

# Loose ceiling: well above any realistic Hungarian-on-150x150 runtime on
# CI hardware.  Tightening this would turn a regression guard into a
# flaky benchmark.
_WIDE_TIME_BUDGET_SECONDS = 2.0


def _wide_dict(n: int = 150) -> dict[str, int]:
    return {f"key_{i}": i for i in range(n)}


def test_wide_identical_scores_one() -> None:
    """A 150-key object compared to itself must score exactly 1.0."""
    payload = _wide_dict()
    result = compare(payload, payload)
    assert result.similarity_score == 1.0


def test_wide_identical_under_time_budget() -> None:
    """Identical wide objects must complete well under the budget.

    Catches obvious cubic / quadratic regressions in the matching path.
    """
    payload = _wide_dict()
    start = time.perf_counter()
    compare(payload, payload)
    elapsed = time.perf_counter() - start
    assert elapsed < _WIDE_TIME_BUDGET_SECONDS, (
        f"wide identical compare took {elapsed:.3f}s, "
        f"budget {_WIDE_TIME_BUDGET_SECONDS:.1f}s"
    )


def test_wide_with_renames_scores_high_but_not_one() -> None:
    """Renaming 5 of 150 keys yields a high-but-not-perfect score.

    Pins the observed score range so future tuning of the matcher is a
    deliberate choice.  The exact value (0.988…) is checked loosely to
    keep the test robust to small algorithmic refinements.
    """
    left = _wide_dict()
    right = dict(left)
    for i in range(5):
        right[f"renamed_{i}"] = right.pop(f"key_{i}")

    result = compare(left, right)
    assert 0.9 < result.similarity_score < 1.0


def test_wide_with_renames_under_time_budget() -> None:
    """Time budget also applies to the rename case (full Hungarian path)."""
    left = _wide_dict()
    right = dict(left)
    for i in range(5):
        right[f"renamed_{i}"] = right.pop(f"key_{i}")

    start = time.perf_counter()
    compare(left, right)
    elapsed = time.perf_counter() - start
    assert elapsed < _WIDE_TIME_BUDGET_SECONDS, (
        f"wide-with-renames compare took {elapsed:.3f}s, "
        f"budget {_WIDE_TIME_BUDGET_SECONDS:.1f}s"
    )


def _wide_dict_with_deep_subtree(n: int, deep_levels: int, leaf_value: str) -> dict:
    """Build a wide dict (n flat KEY+SCALAR pairs) plus one deeply-nested KEY.

    The deep KEY's value is a chain of nested OBJECTs ``deep_levels`` deep,
    ending in a leaf scalar that callers can vary to inject a single-leaf
    difference.
    """
    payload: dict = {f"key_{i}": i for i in range(n)}
    inner: dict = {"leaf": leaf_value}
    for _ in range(deep_levels):
        inner = {"inner": inner}
    payload["deep"] = inner
    return payload


def test_wide_object_with_single_deep_leaf_change_scores_near_one() -> None:
    """Wave-7 contract (audit C6): one deep leaf differs in a ~100-key object.

    Before the wave-7 fix, ``len(children)``-based normalisation drove this
    case below 0.1 (the matched-pair raw distance from the deep subtree
    exceeded the OBJECT's child count and clipped to similarity 0).  The
    Zhang-Shasha denominator (sum of children subtree sizes) keeps the
    score in the [0.95, 1.0) band — almost-but-not-perfect, matching the
    intuition that "99 of 100 keys identical, one deep leaf differs"
    should not be a structural break.
    """
    left = _wide_dict_with_deep_subtree(n=100, deep_levels=10, leaf_value="foo")
    right = _wide_dict_with_deep_subtree(n=100, deep_levels=10, leaf_value="bar")

    result = compare(left, right)
    assert 0.95 <= result.similarity_score < 1.0, (
        f"single-deep-leaf-change scored {result.similarity_score:.4f}, "
        "expected [0.95, 1.0)"
    )
