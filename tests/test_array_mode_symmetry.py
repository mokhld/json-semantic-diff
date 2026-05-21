"""Symmetry tests for ORDERED / UNORDERED / AUTO array comparison modes.

The audit (H9) flagged that the existing symmetry coverage didn't exercise
AUTO or UNORDERED.  Wave 2's hypothesis-based property test xfailed AUTO as
a known concern.  These deterministic tests lock down concrete shapes that
ARE symmetric, so any future regression is caught immediately.  If an AUTO
shape ever flips asymmetric, mark only THAT parametrize entry as
``pytest.mark.xfail`` rather than disabling the whole test.
"""

from __future__ import annotations

from typing import Any

import pytest

from json_semantic_diff import ArrayComparisonMode, STEDConfig, compare

# Tight tolerance — we expect bit-exact equality from the deterministic
# algorithm, but allow a sub-nano slack for floating-point drift.
TOL = 1e-9

MODES: list[ArrayComparisonMode] = [
    ArrayComparisonMode.ORDERED,
    ArrayComparisonMode.UNORDERED,
    ArrayComparisonMode.AUTO,
]

# Each case is (label, left, right).  We assert
# compare(a, b, mode).similarity_score == compare(b, a, mode).similarity_score
# for every mode in MODES.
SHAPES: list[tuple[str, Any, Any]] = [
    ("identical_scalars", [1, 2, 3], [1, 2, 3]),
    ("identical_strings", ["a", "b", "c"], ["a", "b", "c"]),
    ("identical_empty", [], []),
    ("empty_vs_three_scalars", [], [1, 2, 3]),
    ("empty_vs_three_objects", [], [{"x": 1}, {"x": 2}, {"x": 3}]),
    ("reorder_scalars", [1, 2, 3], [3, 2, 1]),
    ("reorder_strings", ["a", "b", "c"], ["c", "b", "a"]),
    ("objects_reorder", [{"x": 1}, {"x": 2}], [{"x": 2}, {"x": 1}]),
    ("objects_same_order", [{"x": 1}, {"x": 2}], [{"x": 1}, {"x": 2}]),
    ("mixed_identical", [1, "two", {"k": 1}, None], [1, "two", {"k": 1}, None]),
    ("mixed_reorder", [1, "two", {"k": 1}], [{"k": 1}, "two", 1]),
    ("mixed_vs_scalars", [1, {"k": 1}, 3], [1, 2, 3]),
    ("scalars_vs_objects", [1, 2, 3], [{"k": 1}, {"k": 2}, {"k": 3}]),
    ("nested_arrays", [[1, 2], [3, 4]], [[3, 4], [1, 2]]),
    ("one_extra_element", [1, 2, 3], [1, 2, 3, 4]),
]


def _params() -> list[Any]:
    """Cartesian product of MODES x SHAPES as a flat parametrize list.

    Each row carries enough metadata for an xfail decision on a single
    (mode, shape) cell without affecting the rest of the matrix.
    """
    rows: list[Any] = []
    for mode in MODES:
        for label, left, right in SHAPES:
            rows.append(
                pytest.param(
                    mode,
                    left,
                    right,
                    id=f"{mode.name.lower()}-{label}",
                )
            )
    return rows


@pytest.mark.parametrize(("mode", "left", "right"), _params())
def test_array_mode_symmetry(
    mode: ArrayComparisonMode,
    left: Any,
    right: Any,
) -> None:
    """compare(a, b) and compare(b, a) score within ``TOL`` for every mode.

    All cells currently pass — there are no observed AUTO-asymmetry shapes
    among these deterministic inputs.  If a future change introduces one,
    convert the offending row to a ``pytest.mark.xfail(strict=False)`` with
    a reference to audit H9 rather than weakening the whole matrix.
    """
    config = STEDConfig(array_comparison_mode=mode)
    forward = compare(left, right, config=config).similarity_score
    reverse = compare(right, left, config=config).similarity_score
    assert forward == pytest.approx(reverse, abs=TOL), (
        f"asymmetric under {mode.name}: "
        f"compare(a, b)={forward}, compare(b, a)={reverse}"
    )
