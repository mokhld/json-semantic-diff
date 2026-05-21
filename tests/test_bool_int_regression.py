"""Regression tests for bool/int conflation (audit H2, fixed in Wave 1).

Python treats ``bool`` as a subclass of ``int`` — ``True == 1`` and
``False == 0``.  The Wave 1 fix makes the comparator treat ``bool`` as its
own type, so ``{"flag": True}`` does NOT score identical to ``{"flag": 1}``.

The bool-exclusion guard must hold regardless of ``type_coercion``, because
the fix lives below the coercion layer (bools are deliberately not coerced
to/from ints).  These tests pin that contract at top-level, nested-path, and
inside-array locations.
"""

from __future__ import annotations

import pytest

from json_semantic_diff import STEDConfig, compare

# Both default and type_coercion=True must observe the same bool-vs-int rule.
CONFIGS: list[tuple[str, STEDConfig | None]] = [
    ("default", None),
    ("type_coercion", STEDConfig(type_coercion=True)),
]


@pytest.mark.parametrize(("label", "config"), CONFIGS)
class TestBoolIntRegression:
    """Bool must not be conflated with int (or string) under any config."""

    # ------------------------------------------------------------------
    # Identity (sanity)
    # ------------------------------------------------------------------

    def test_identical_true_scores_one(
        self, label: str, config: STEDConfig | None
    ) -> None:
        """Two ``True`` values are identical under any config."""
        result = compare({"flag": True}, {"flag": True}, config=config)
        assert result.similarity_score == pytest.approx(1.0)

    def test_identical_false_scores_one(
        self, label: str, config: STEDConfig | None
    ) -> None:
        """Two ``False`` values are identical under any config."""
        result = compare({"flag": False}, {"flag": False}, config=config)
        assert result.similarity_score == pytest.approx(1.0)

    # ------------------------------------------------------------------
    # Main regression: bool vs int at top level
    # ------------------------------------------------------------------

    def test_true_not_equal_to_one(self, label: str, config: STEDConfig | None) -> None:
        """``True`` must NOT be conflated with ``1`` (audit H2)."""
        result = compare({"flag": True}, {"flag": 1}, config=config)
        assert result.similarity_score < 1.0

    def test_false_not_equal_to_zero(
        self, label: str, config: STEDConfig | None
    ) -> None:
        """``False`` must NOT be conflated with ``0`` (sister case)."""
        result = compare({"flag": False}, {"flag": 0}, config=config)
        assert result.similarity_score < 1.0

    # ------------------------------------------------------------------
    # Bool vs string: no coercion either way
    # ------------------------------------------------------------------

    def test_true_not_equal_to_string_true(
        self, label: str, config: STEDConfig | None
    ) -> None:
        """``True`` is not coerced to/from the string ``"true"``."""
        result = compare({"flag": True}, {"flag": "true"}, config=config)
        assert result.similarity_score < 1.0

    def test_false_not_equal_to_string_false(
        self, label: str, config: STEDConfig | None
    ) -> None:
        """``False`` is not coerced to/from the string ``"false"``."""
        result = compare({"flag": False}, {"flag": "false"}, config=config)
        assert result.similarity_score < 1.0

    # ------------------------------------------------------------------
    # Nested paths
    # ------------------------------------------------------------------

    def test_nested_true_not_equal_to_one(
        self, label: str, config: STEDConfig | None
    ) -> None:
        """Bool/int guard holds inside a nested object subtree."""
        left = {"user": {"flag": True}}
        right = {"user": {"flag": 1}}
        result = compare(left, right, config=config)
        assert result.similarity_score < 1.0

    def test_nested_false_not_equal_to_zero(
        self, label: str, config: STEDConfig | None
    ) -> None:
        """Bool/int guard holds for ``False`` at a nested path."""
        left = {"user": {"flag": False}}
        right = {"user": {"flag": 0}}
        result = compare(left, right, config=config)
        assert result.similarity_score < 1.0

    # ------------------------------------------------------------------
    # Inside arrays
    # ------------------------------------------------------------------

    def test_array_true_not_equal_to_one(
        self, label: str, config: STEDConfig | None
    ) -> None:
        """Bool/int guard holds when the values appear inside an array."""
        result = compare([True], [1], config=config)
        assert result.similarity_score < 1.0

    def test_array_false_not_equal_to_zero(
        self, label: str, config: STEDConfig | None
    ) -> None:
        """Bool/int guard holds for ``False`` vs ``0`` inside arrays."""
        result = compare([False], [0], config=config)
        assert result.similarity_score < 1.0
