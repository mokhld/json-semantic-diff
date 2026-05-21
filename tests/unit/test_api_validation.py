"""Unit tests for the validation fixes in the public API and comparator.

Covers:
- M3: is_equivalent() raises ValueError on out-of-range threshold.
- D3: STEDComparator.compare() raises TypeError on non-JSON inputs.
- H10: compare() does not mutate its inputs (even with null_equals_missing=False).
"""

from __future__ import annotations

import copy
import datetime as dt

import pytest

from json_semantic_diff import STEDConfig, compare, is_equivalent
from json_semantic_diff.comparator import STEDComparator

# ---------------------------------------------------------------------------
# M3 — is_equivalent threshold validation
# ---------------------------------------------------------------------------


class TestIsEquivalentThresholdValidation:
    def test_threshold_above_1_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            is_equivalent({"a": 1}, {"a": 1}, threshold=1.5)

    def test_threshold_below_0_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            is_equivalent({"a": 1}, {"a": 1}, threshold=-0.1)

    def test_threshold_0_is_allowed(self) -> None:
        assert is_equivalent({"a": 1}, {"a": 1}, threshold=0.0) is True

    def test_threshold_1_is_allowed(self) -> None:
        assert is_equivalent({"a": 1}, {"a": 1}, threshold=1.0) is True

    def test_threshold_inside_range_is_allowed(self) -> None:
        assert isinstance(is_equivalent({"a": 1}, {"a": 1}, threshold=0.5), bool)


# ---------------------------------------------------------------------------
# D3 — top-level JSON type validation
# ---------------------------------------------------------------------------


class _CustomObject:
    """A non-JSON class used to verify TypeError surfaces at the API boundary."""


class TestCompareTopLevelTypeValidation:
    def test_datetime_left_raises_type_error(self) -> None:
        cmp = STEDComparator()
        with pytest.raises(TypeError, match="JSON values"):
            cmp.compare(dt.datetime(2024, 1, 1), {"a": 1})

    def test_set_right_raises_type_error(self) -> None:
        cmp = STEDComparator()
        with pytest.raises(TypeError, match="JSON values"):
            cmp.compare({"a": 1}, {1, 2, 3})

    def test_custom_class_instance_raises_type_error(self) -> None:
        cmp = STEDComparator()
        with pytest.raises(TypeError, match="JSON values"):
            cmp.compare(_CustomObject(), {"a": 1})

    def test_error_message_names_the_offending_argument(self) -> None:
        cmp = STEDComparator()
        with pytest.raises(TypeError, match="right"):
            cmp.compare({"a": 1}, _CustomObject())

    def test_top_level_dict_passes(self) -> None:
        cmp = STEDComparator()
        cmp.compare({"a": 1}, {"a": 1})  # must not raise

    def test_top_level_list_passes(self) -> None:
        cmp = STEDComparator()
        cmp.compare([1, 2], [1, 2])  # must not raise

    def test_top_level_string_passes(self) -> None:
        cmp = STEDComparator()
        cmp.compare("hi", "hi")  # must not raise

    def test_top_level_none_passes(self) -> None:
        cmp = STEDComparator()
        cmp.compare(None, None)  # must not raise

    def test_top_level_bool_passes(self) -> None:
        cmp = STEDComparator()
        cmp.compare(True, False)  # must not raise

    def test_top_level_int_passes(self) -> None:
        cmp = STEDComparator()
        cmp.compare(1, 2)  # must not raise

    def test_top_level_float_passes(self) -> None:
        cmp = STEDComparator()
        cmp.compare(1.0, 2.0)  # must not raise


# ---------------------------------------------------------------------------
# H10 — compare() never mutates its inputs
# ---------------------------------------------------------------------------


class TestCompareDoesNotMutateInputs:
    def test_default_config_does_not_mutate_left(self) -> None:
        left = {"a": 1, "b": {"c": 2}}
        original = copy.deepcopy(left)
        compare(left, {"a": 1})
        assert left == original

    def test_default_config_does_not_mutate_right(self) -> None:
        right = {"a": 1, "b": [1, 2, 3]}
        original = copy.deepcopy(right)
        compare({"a": 1}, right)
        assert right == original

    def test_null_equals_missing_does_not_mutate_left(self) -> None:
        left = {"a": 1, "b": None, "c": {"d": None, "e": 5}}
        original = copy.deepcopy(left)
        compare(left, {"a": 1}, config=STEDConfig(null_equals_missing=True))
        assert left == original

    def test_null_equals_missing_does_not_mutate_right(self) -> None:
        right = {"a": 1, "b": None}
        original = copy.deepcopy(right)
        compare({"a": 1}, right, config=STEDConfig(null_equals_missing=True))
        assert right == original

    def test_nested_list_in_dict_is_not_mutated(self) -> None:
        left = {"items": [{"x": 1, "y": None}, {"x": 2}]}
        original = copy.deepcopy(left)
        compare(left, {"items": []}, config=STEDConfig(null_equals_missing=True))
        assert left == original

    def test_input_identity_not_returned_from_preprocess(self) -> None:
        """_preprocess must always return a fresh object for containers."""
        cmp = STEDComparator(config=STEDConfig(null_equals_missing=False))
        original = {"a": 1, "b": [1, 2]}
        # _preprocess is internal but documenting the contract here.
        out = cmp._preprocess(original)
        assert out == original
        assert out is not original
        assert out["b"] is not original["b"]
