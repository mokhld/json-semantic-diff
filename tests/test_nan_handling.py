"""NaN and inf scalar handling (audit gap T2).

These tests pin the *current* documented behaviour of ``compare`` when JSON
inputs contain IEEE-754 special values.  They are intentionally
characterisation tests — if the behaviour changes we want to know about it
deliberately.

Background
----------
Python's equality semantics for ``float`` are surprising at the edges:

* ``float('nan') == float('nan')`` is **False** by IEEE-754 design.
* ``float('inf') == float('inf')`` is **True**.

The library inherits these semantics for scalar leaves, which means two
identical-looking NaN-containing documents currently score < 1.0.  Whether
to special-case NaN equality is a product decision and not yet resolved;
this test file documents the status quo so any future change is an
intentional choice.
"""

from __future__ import annotations

import math

from json_semantic_diff import compare


def test_nan_vs_nan_does_not_score_one() -> None:
    """``nan != nan`` propagates into the leaf comparison.

    Two ``{"x": nan}`` objects look identical to the eye but the value
    comparator sees them as unequal.  Pinning the observed score so a
    behaviour change requires touching this test deliberately.
    """
    result = compare({"x": float("nan")}, {"x": float("nan")})
    # NaN equality is unusual in Python; this is the documented behaviour
    # for now (no special-casing of NaN inside the leaf comparator).
    # Audit C6 (wave 7): with the Zhang-Shasha subtree-size denominator
    # the KEY-pair raw distance (0.5 for the NaN content mismatch) is
    # divided by 2 (KEY + SCALAR), giving a similarity floor of 0.75.
    assert result.similarity_score == 0.75


def test_inf_vs_inf_scores_one() -> None:
    """``inf == inf`` in Python, so identical infinity values match."""
    result = compare({"x": float("inf")}, {"x": float("inf")})
    assert result.similarity_score == 1.0


def test_neg_inf_vs_neg_inf_scores_one() -> None:
    """``-inf == -inf`` in Python, mirroring the +inf case."""
    result = compare({"x": float("-inf")}, {"x": float("-inf")})
    assert result.similarity_score == 1.0


def test_nan_vs_number_does_not_score_one() -> None:
    """A NaN paired with a real number must not be considered identical."""
    result = compare({"x": float("nan")}, {"x": 1.0})
    assert result.similarity_score < 1.0


def test_root_nan_pin() -> None:
    """Pin the root-scalar NaN comparison behaviour as well."""
    result = compare(float("nan"), float("nan"))
    assert result.similarity_score == 0.5


def test_root_inf_pin() -> None:
    """Root-scalar identical infinities score 1.0."""
    result = compare(float("inf"), float("inf"))
    assert result.similarity_score == 1.0


def test_score_is_always_finite_for_nan_input() -> None:
    """A NaN inside the input must not poison the output similarity score.

    The similarity score is a public, advertised float in ``[0, 1]``.  No
    matter how exotic the inputs are (NaN, +inf, -inf at multiple depths),
    the returned ``similarity_score`` must remain finite and inside the
    documented range.  This guards against accidental arithmetic that
    would let a NaN leaf-score bubble all the way up.
    """
    nasty = {
        "scalar_nan": float("nan"),
        "scalar_pos_inf": float("inf"),
        "scalar_neg_inf": float("-inf"),
        "nested": {
            "deep_nan": float("nan"),
            "list_with_nan": [1.0, float("nan"), float("inf"), 2.0],
        },
    }
    benign = {
        "scalar_nan": 0.0,
        "scalar_pos_inf": 1.0,
        "scalar_neg_inf": -1.0,
        "nested": {
            "deep_nan": 3.14,
            "list_with_nan": [1.0, 2.0, 3.0, 4.0],
        },
    }
    result = compare(nasty, benign)
    assert math.isfinite(result.similarity_score)
    assert 0.0 <= result.similarity_score <= 1.0


def test_score_is_always_finite_when_both_sides_nan() -> None:
    """Two NaN-laden inputs must also produce a finite score."""
    payload = {
        "a": float("nan"),
        "b": float("inf"),
        "c": [float("nan"), float("nan")],
    }
    result = compare(payload, payload)
    assert math.isfinite(result.similarity_score)
    assert 0.0 <= result.similarity_score <= 1.0
