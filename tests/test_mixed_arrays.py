"""Mixed-type array tests for AUTO mode and beyond.

Real-world JSON arrays often mix scalar, object, and null entries, e.g.
``[1, "two", {"three": 3}, None]``.  These tests pin the public-API behaviour
of :func:`compare` on such inputs, focusing on AUTO mode's heuristic and on
ORDERED-mode null-vs-missing semantics that an earlier audit flagged as
under-tested.
"""

from __future__ import annotations

import pytest

from json_semantic_diff import (
    ArrayComparisonMode,
    STEDConfig,
    compare,
)


class TestMixedArrays:
    """Tests covering heterogeneous arrays under the default config."""

    def test_identical_mixed_array_scores_one(self) -> None:
        """Two identical mixed-type arrays score 1.0."""
        left = {"items": [1, "two", {"three": 3}, None]}
        right = {"items": [1, "two", {"three": 3}, None]}
        result = compare(left, right)
        assert result.similarity_score == pytest.approx(1.0)

    def test_mixed_array_one_value_differs_scores_below_one(self) -> None:
        """A single differing element drops the score below 1.0."""
        left = {"items": [1, "two", {"three": 3}, None]}
        right = {"items": [1, "two", {"three": 4}, None]}
        result = compare(left, right)
        assert result.similarity_score < 1.0

    def test_heterogeneous_vs_all_scalar_pinned(self) -> None:
        """Comparing a mixed array against an all-scalar one is pinned.

        Under default (ORDERED) mode, three of four positions match in shape;
        only the ``{"three": 3}`` vs ``3`` slot is a type mismatch.

        Audit C6 (wave 7): under the previous ``len(children)``-based
        normaliser this binary-collapsed to ~0.0.  With the Zhang-Shasha
        subtree-size denominator the score reflects the actual amount of
        matching structure — pinned here so future heuristic changes are
        noticed.

        Audit I4 (wave 8): lambda_unmatched bumped 0.1 → 0.5, so the
        size-diff term (12 vs 10 here) contributes ~5x more to the
        numerator and the score moves down from 0.7333 to 0.6667.
        Drift-toward-correct: the array's structural asymmetry now bites
        proportionally to the size diff.
        """
        left = {"items": [1, "two", {"three": 3}, None]}
        right = {"items": [1, 2, 3, 4]}
        result = compare(left, right)
        assert result.similarity_score == pytest.approx(0.6667, abs=0.01)

    def test_auto_with_object_picks_ordered(self) -> None:
        """AUTO mode falls back to ORDERED when any element is an object.

        We verify by reordering: in UNORDERED, ``[1,{"x":1},3]`` vs
        ``[3,{"x":1},1]`` would score 1.0; in ORDERED it does not.  If AUTO
        picks ORDERED (per the documented heuristic), the score is < 1.0.
        """
        auto = STEDConfig(array_comparison_mode=ArrayComparisonMode.AUTO)
        unord = STEDConfig(array_comparison_mode=ArrayComparisonMode.UNORDERED)

        left = [1, {"x": 1}, 3]
        right = [3, {"x": 1}, 1]

        auto_score = compare(left, right, config=auto).similarity_score
        unord_score = compare(left, right, config=unord).similarity_score

        assert unord_score == pytest.approx(1.0)
        assert auto_score < 1.0

    def test_empty_mixed_arrays_score_one(self) -> None:
        """Two empty arrays compare as identical regardless of mode."""
        result = compare({"items": []}, {"items": []})
        assert result.similarity_score == pytest.approx(1.0)

    def test_null_differs_from_missing_index_in_ordered(self) -> None:
        """Under ORDERED mode, ``[None]`` differs from ``[]``.

        ``None`` occupies a positional slot; an empty array has no slot at
        index 0, so the score should drop below 1.0.
        """
        ord_cfg = STEDConfig(array_comparison_mode=ArrayComparisonMode.ORDERED)
        result = compare([None], [], config=ord_cfg)
        assert result.similarity_score < 1.0

    def test_null_inside_mixed_array_changes_score(self) -> None:
        """A None entry that aligns positionally with a non-null differs."""
        ord_cfg = STEDConfig(array_comparison_mode=ArrayComparisonMode.ORDERED)
        left = [1, None, 3]
        right = [1, 2, 3]
        result = compare(left, right, config=ord_cfg)
        assert result.similarity_score < 1.0

    def test_mixed_array_with_extra_element_scores_below_one(self) -> None:
        """Adding an element to a mixed array drops the score below 1.0."""
        left = {"items": [1, "two", {"three": 3}, None]}
        right = {"items": [1, "two", {"three": 3}, None, "extra"]}
        result = compare(left, right)
        assert result.similarity_score < 1.0
