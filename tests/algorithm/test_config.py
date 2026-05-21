"""Tests for STEDConfig frozen dataclass and ArrayComparisonMode StrEnum.

Covers:
- Default values (w_s=0.5, w_c=0.5, lambda_unmatched=0.1, mode=ORDERED)
- Custom weight construction
- Immutability (FrozenInstanceError on assignment)
- Validation: w_s + w_c must sum to 1.0 within tolerance
- Validation: weights must be in [0, 1]
- Validation: lambda_unmatched must be >= 0.0
- ArrayComparisonMode has exactly three values: ordered, unordered, auto
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from json_semantic_diff.algorithm.config import ArrayComparisonMode, STEDConfig

# ---------------------------------------------------------------------------
# ArrayComparisonMode
# ---------------------------------------------------------------------------


class TestArrayComparisonMode:
    def test_has_exactly_three_members(self) -> None:
        members = list(ArrayComparisonMode)
        assert len(members) == 3

    def test_ordered_value(self) -> None:
        assert ArrayComparisonMode.ORDERED == "ordered"

    def test_unordered_value(self) -> None:
        assert ArrayComparisonMode.UNORDERED == "unordered"

    def test_auto_value(self) -> None:
        assert ArrayComparisonMode.AUTO == "auto"

    def test_is_str_subclass(self) -> None:
        # StrEnum members are strings
        assert isinstance(ArrayComparisonMode.ORDERED, str)

    def test_all_expected_members_exist(self) -> None:
        names = {m.name for m in ArrayComparisonMode}
        assert names == {"ORDERED", "UNORDERED", "AUTO"}


# ---------------------------------------------------------------------------
# STEDConfig — default construction
# ---------------------------------------------------------------------------


class TestSTEDConfigDefaults:
    def test_default_w_s(self) -> None:
        config = STEDConfig()
        assert config.w_s == 0.5

    def test_default_w_c(self) -> None:
        config = STEDConfig()
        assert config.w_c == 0.5

    def test_default_lambda_unmatched(self) -> None:
        # audit I4 (wave 8): default bumped 0.1 → 0.5 so the unmatched
        # penalty bites proportionally to the subtree-size diff that
        # wave-7 (C6) introduced into the normaliser denominator.
        config = STEDConfig()
        assert config.lambda_unmatched == 0.5

    def test_default_array_comparison_mode(self) -> None:
        config = STEDConfig()
        assert config.array_comparison_mode == ArrayComparisonMode.ORDERED

    def test_weights_sum_to_one_by_default(self) -> None:
        config = STEDConfig()
        assert abs(config.w_s + config.w_c - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# STEDConfig — custom construction
# ---------------------------------------------------------------------------


class TestSTEDConfigCustom:
    def test_custom_weights(self) -> None:
        config = STEDConfig(w_s=0.7, w_c=0.3)
        assert config.w_s == pytest.approx(0.7)
        assert config.w_c == pytest.approx(0.3)

    def test_custom_lambda_unmatched(self) -> None:
        config = STEDConfig(w_s=0.6, w_c=0.4, lambda_unmatched=0.25)
        assert config.lambda_unmatched == pytest.approx(0.25)

    def test_custom_array_comparison_mode_unordered(self) -> None:
        config = STEDConfig(array_comparison_mode=ArrayComparisonMode.UNORDERED)
        assert config.array_comparison_mode == ArrayComparisonMode.UNORDERED

    def test_custom_array_comparison_mode_auto(self) -> None:
        config = STEDConfig(array_comparison_mode=ArrayComparisonMode.AUTO)
        assert config.array_comparison_mode == ArrayComparisonMode.AUTO

    def test_extreme_weights_zero_one(self) -> None:
        # w_s=0.0, w_c=1.0 is valid (sums to 1.0)
        config = STEDConfig(w_s=0.0, w_c=1.0)
        assert config.w_s == pytest.approx(0.0)
        assert config.w_c == pytest.approx(1.0)

    def test_extreme_weights_one_zero(self) -> None:
        # w_s=1.0, w_c=0.0 is valid (sums to 1.0)
        config = STEDConfig(w_s=1.0, w_c=0.0)
        assert config.w_s == pytest.approx(1.0)
        assert config.w_c == pytest.approx(0.0)

    def test_lambda_zero_is_valid(self) -> None:
        config = STEDConfig(lambda_unmatched=0.0)
        assert config.lambda_unmatched == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# STEDConfig — immutability
# ---------------------------------------------------------------------------


class TestSTEDConfigImmutability:
    def test_frozen_w_s_raises(self) -> None:
        config = STEDConfig()
        with pytest.raises(FrozenInstanceError):
            config.w_s = 0.8  # type: ignore[misc]

    def test_frozen_w_c_raises(self) -> None:
        config = STEDConfig()
        with pytest.raises(FrozenInstanceError):
            config.w_c = 0.2  # type: ignore[misc]

    def test_frozen_lambda_raises(self) -> None:
        config = STEDConfig()
        with pytest.raises(FrozenInstanceError):
            config.lambda_unmatched = 0.5  # type: ignore[misc]

    def test_frozen_mode_raises(self) -> None:
        config = STEDConfig()
        with pytest.raises(FrozenInstanceError):
            config.array_comparison_mode = ArrayComparisonMode.AUTO  # type: ignore[misc]

    def test_is_hashable(self) -> None:
        # Frozen dataclasses should be hashable
        config = STEDConfig()
        assert isinstance(hash(config), int)

    def test_equality(self) -> None:
        c1 = STEDConfig()
        c2 = STEDConfig()
        assert c1 == c2

    def test_inequality_different_weights(self) -> None:
        c1 = STEDConfig(w_s=0.7, w_c=0.3)
        c2 = STEDConfig(w_s=0.6, w_c=0.4)
        assert c1 != c2


# ---------------------------------------------------------------------------
# STEDConfig — validation errors
# ---------------------------------------------------------------------------


class TestSTEDConfigValidation:
    def test_weights_not_summing_to_one_raises(self) -> None:
        with pytest.raises(ValueError, match="sum to 1"):
            STEDConfig(w_s=0.8, w_c=0.8)

    def test_weights_summing_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="sum to 1"):
            STEDConfig(w_s=0.3, w_c=0.3)

    def test_negative_w_s_raises(self) -> None:
        with pytest.raises(ValueError):
            STEDConfig(w_s=-0.1, w_c=1.1)

    def test_w_s_above_one_raises(self) -> None:
        with pytest.raises(ValueError):
            STEDConfig(w_s=1.1, w_c=-0.1)

    def test_negative_w_c_raises(self) -> None:
        with pytest.raises(ValueError):
            STEDConfig(w_s=1.1, w_c=-0.1)

    def test_negative_lambda_raises(self) -> None:
        with pytest.raises(ValueError):
            STEDConfig(lambda_unmatched=-0.01)

    def test_tolerance_boundary_is_strict(self) -> None:
        # Just inside tolerance — should succeed (fp arithmetic)
        config = STEDConfig(w_s=0.3333333333333333, w_c=0.6666666666666667)
        assert abs(config.w_s + config.w_c - 1.0) < 1e-9
