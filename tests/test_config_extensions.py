"""Tests for STEDConfig extension fields: type_coercion and null_equals_missing.

Covers:
- Default values: type_coercion=False, null_equals_missing=False
- Successful construction with each new field set to True
- Frozen (immutable) enforcement on new fields
- Backward compatibility: STEDConfig() behaviour identical to before
- Type coercion integration with _content_distance (SCALAR nodes)
  - type_coercion=True: "123" vs 123 -> 0.0
  - type_coercion=False (default): "123" vs 123 -> 1.0
  - Float coercion: "3.14" vs 3.14 -> 0.0
  - Non-numeric: "hello" vs 42 -> 1.0 (coercion fails, falls back)
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from json_semantic_diff.algorithm.config import ArrayComparisonMode, STEDConfig
from json_semantic_diff.algorithm.costs import _content_distance
from json_semantic_diff.tree.nodes import NodeType, TreeNode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_scalar(value: object, path: str = "/val") -> TreeNode:
    """Create a SCALAR TreeNode with the given value."""
    return TreeNode(
        node_type=NodeType.SCALAR,
        label=str(value),
        path=path,
        value=value,
    )


# ---------------------------------------------------------------------------
# New field defaults
# ---------------------------------------------------------------------------


class TestSTEDConfigNewFieldDefaults:
    def test_type_coercion_default_false(self) -> None:
        config = STEDConfig()
        assert config.type_coercion is False

    def test_null_equals_missing_default_false(self) -> None:
        config = STEDConfig()
        assert config.null_equals_missing is False

    def test_both_defaults_false(self) -> None:
        config = STEDConfig()
        assert not config.type_coercion
        assert not config.null_equals_missing


# ---------------------------------------------------------------------------
# Construction with new fields
# ---------------------------------------------------------------------------


class TestSTEDConfigNewFieldConstruction:
    def test_type_coercion_true_constructs(self) -> None:
        config = STEDConfig(type_coercion=True)
        assert config.type_coercion is True

    def test_null_equals_missing_true_constructs(self) -> None:
        config = STEDConfig(null_equals_missing=True)
        assert config.null_equals_missing is True

    def test_both_new_fields_true_constructs(self) -> None:
        config = STEDConfig(type_coercion=True, null_equals_missing=True)
        assert config.type_coercion is True
        assert config.null_equals_missing is True

    def test_new_fields_combined_with_existing(self) -> None:
        config = STEDConfig(
            w_s=0.7,
            w_c=0.3,
            type_coercion=True,
            null_equals_missing=True,
            array_comparison_mode=ArrayComparisonMode.UNORDERED,
        )
        assert config.w_s == pytest.approx(0.7)
        assert config.w_c == pytest.approx(0.3)
        assert config.type_coercion is True
        assert config.null_equals_missing is True
        assert config.array_comparison_mode == ArrayComparisonMode.UNORDERED


# ---------------------------------------------------------------------------
# Frozen enforcement on new fields
# ---------------------------------------------------------------------------


class TestSTEDConfigNewFieldsFrozen:
    def test_type_coercion_is_frozen(self) -> None:
        config = STEDConfig(type_coercion=True)
        with pytest.raises(FrozenInstanceError):
            config.type_coercion = False  # type: ignore[misc]

    def test_null_equals_missing_is_frozen(self) -> None:
        config = STEDConfig(null_equals_missing=True)
        with pytest.raises(FrozenInstanceError):
            config.null_equals_missing = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestSTEDConfigBackwardCompatibility:
    def test_default_w_s_unchanged(self) -> None:
        config = STEDConfig()
        assert config.w_s == 0.5

    def test_default_w_c_unchanged(self) -> None:
        config = STEDConfig()
        assert config.w_c == 0.5

    def test_default_lambda_unmatched(self) -> None:
        # audit I4 (wave 8): default re-calibrated from 0.1 → 0.5 to
        # match the wave-7 subtree-size denominator.  Test renamed off
        # "...unchanged" because the value deliberately changed.
        config = STEDConfig()
        assert config.lambda_unmatched == pytest.approx(0.5)

    def test_default_array_comparison_mode_unchanged(self) -> None:
        config = STEDConfig()
        assert config.array_comparison_mode == ArrayComparisonMode.ORDERED

    def test_weights_still_sum_to_one(self) -> None:
        config = STEDConfig()
        assert abs(config.w_s + config.w_c - 1.0) < 1e-9

    def test_still_hashable(self) -> None:
        config = STEDConfig()
        assert isinstance(hash(config), int)

    def test_equality_still_works(self) -> None:
        c1 = STEDConfig()
        c2 = STEDConfig()
        assert c1 == c2

    def test_custom_weights_still_validate(self) -> None:
        config = STEDConfig(w_s=0.8, w_c=0.2)
        assert config.w_s == pytest.approx(0.8)
        assert config.w_c == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Type coercion integration with _content_distance
# ---------------------------------------------------------------------------


class TestTypeCoercionInContentDistance:
    def test_string_int_equal_with_type_coercion(self) -> None:
        """SCALAR("123") vs SCALAR(123) with type_coercion=True -> 0.0."""
        node_a = make_scalar("123")
        node_b = make_scalar(123)
        config = STEDConfig(type_coercion=True)
        assert _content_distance(node_a, node_b, config) == pytest.approx(0.0)

    def test_string_int_different_without_type_coercion(self) -> None:
        """SCALAR("123") vs SCALAR(123) with type_coercion=False -> 1.0."""
        node_a = make_scalar("123")
        node_b = make_scalar(123)
        config = STEDConfig()  # type_coercion=False by default
        assert _content_distance(node_a, node_b, config) == pytest.approx(1.0)

    def test_float_string_equal_with_type_coercion(self) -> None:
        """SCALAR("3.14") vs SCALAR(3.14) with type_coercion=True -> 0.0."""
        node_a = make_scalar("3.14")
        node_b = make_scalar(3.14)
        config = STEDConfig(type_coercion=True)
        assert _content_distance(node_a, node_b, config) == pytest.approx(0.0)

    def test_non_numeric_string_vs_int_with_type_coercion(self) -> None:
        """SCALAR("hello") vs SCALAR(42) with type_coercion=True -> 1.0.

        Coercion of "hello" to float raises ValueError, falls back to 1.0.
        """
        node_a = make_scalar("hello")
        node_b = make_scalar(42)
        config = STEDConfig(type_coercion=True)
        assert _content_distance(node_a, node_b, config) == pytest.approx(1.0)

    def test_int_vs_float_coercion_equal(self) -> None:
        """SCALAR(1) vs SCALAR(1.0) -> equal without coercion (Python int/float equality)."""
        node_a = make_scalar(1)
        node_b = make_scalar(1.0)
        config = STEDConfig()  # default
        # Python: 1 == 1.0 is True, so content distance = 0.0 even without coercion
        assert _content_distance(node_a, node_b, config) == pytest.approx(0.0)

    def test_different_numeric_values_with_type_coercion(self) -> None:
        """SCALAR("123") vs SCALAR(456) with type_coercion=True -> 1.0 (different values)."""
        node_a = make_scalar("123")
        node_b = make_scalar(456)
        config = STEDConfig(type_coercion=True)
        assert _content_distance(node_a, node_b, config) == pytest.approx(1.0)

    def test_identical_strings_no_coercion_needed(self) -> None:
        """SCALAR("hello") vs SCALAR("hello") -> 0.0 regardless of type_coercion."""
        node_a = make_scalar("hello")
        node_b = make_scalar("hello")
        for coerce in (True, False):
            config = STEDConfig(type_coercion=coerce)
            assert _content_distance(node_a, node_b, config) == pytest.approx(0.0)

    def test_int_string_reversed_direction(self) -> None:
        """SCALAR(123) vs SCALAR("123") with type_coercion=True -> 0.0 (order symmetric)."""
        node_a = make_scalar(123)
        node_b = make_scalar("123")
        config = STEDConfig(type_coercion=True)
        assert _content_distance(node_a, node_b, config) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ignore_paths configuration
# ---------------------------------------------------------------------------


class TestIgnorePathsDefaults:
    def test_ignore_paths_default_empty_tuple(self) -> None:
        config = STEDConfig()
        assert config.ignore_paths == ()

    def test_ignore_paths_default_is_tuple_type(self) -> None:
        config = STEDConfig()
        assert isinstance(config.ignore_paths, tuple)


class TestIgnorePathsConstruction:
    def test_ignore_paths_single_pattern(self) -> None:
        config = STEDConfig(ignore_paths=("/timestamp",))
        assert config.ignore_paths == ("/timestamp",)

    def test_ignore_paths_multiple_patterns(self) -> None:
        config = STEDConfig(ignore_paths=("/timestamp", "/users/*/id"))
        assert config.ignore_paths == ("/timestamp", "/users/*/id")

    def test_ignore_paths_with_wildcard(self) -> None:
        config = STEDConfig(ignore_paths=("/users/*/id",))
        assert config.ignore_paths == ("/users/*/id",)

    def test_ignore_paths_nested_pattern(self) -> None:
        config = STEDConfig(ignore_paths=("/meta/version/number",))
        assert config.ignore_paths == ("/meta/version/number",)


class TestIgnorePathsFrozen:
    def test_ignore_paths_is_frozen(self) -> None:
        config = STEDConfig(ignore_paths=("/x",))
        with pytest.raises(FrozenInstanceError):
            config.ignore_paths = ("/y",)  # type: ignore[misc]


class TestIgnorePathsValidation:
    def test_missing_leading_slash_raises(self) -> None:
        with pytest.raises(ValueError, match="must start with '/'"):
            STEDConfig(ignore_paths=("timestamp",))

    def test_trailing_slash_raises(self) -> None:
        with pytest.raises(ValueError, match="must not end with '/'"):
            STEDConfig(ignore_paths=("/timestamp/",))

    def test_root_only_raises(self) -> None:
        with pytest.raises(ValueError, match="not a valid target"):
            STEDConfig(ignore_paths=("/",))

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="must start with '/'"):
            STEDConfig(ignore_paths=("",))

    def test_empty_component_raises(self) -> None:
        with pytest.raises(ValueError, match="empty path component"):
            STEDConfig(ignore_paths=("/a//b",))

    def test_non_string_entry_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="must be strings"):
            STEDConfig(ignore_paths=(123,))  # type: ignore[arg-type]

    def test_non_tuple_ignore_paths_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="must be a tuple"):
            STEDConfig(ignore_paths=["/timestamp"])  # type: ignore[arg-type]

    def test_multiple_patterns_one_bad_raises(self) -> None:
        with pytest.raises(ValueError, match="must start with '/'"):
            STEDConfig(ignore_paths=("/ok", "no_slash"))


class TestIgnorePathsHashable:
    def test_config_with_ignore_paths_still_hashable(self) -> None:
        config = STEDConfig(ignore_paths=("/timestamp", "/users/*/id"))
        assert isinstance(hash(config), int)

    def test_equal_ignore_paths_configs_equal(self) -> None:
        c1 = STEDConfig(ignore_paths=("/timestamp",))
        c2 = STEDConfig(ignore_paths=("/timestamp",))
        assert c1 == c2


# ---------------------------------------------------------------------------
# numeric_tolerance configuration (I3)
# ---------------------------------------------------------------------------


class TestNumericToleranceDefaults:
    def test_numeric_tolerance_default_is_zero(self) -> None:
        config = STEDConfig()
        assert config.numeric_tolerance == 0.0

    def test_numeric_tolerance_default_is_float(self) -> None:
        config = STEDConfig()
        assert isinstance(config.numeric_tolerance, float)


class TestNumericToleranceConstruction:
    def test_small_positive_tolerance(self) -> None:
        config = STEDConfig(numeric_tolerance=1e-6)
        assert config.numeric_tolerance == pytest.approx(1e-6)

    def test_large_positive_tolerance(self) -> None:
        config = STEDConfig(numeric_tolerance=100.0)
        assert config.numeric_tolerance == pytest.approx(100.0)

    def test_zero_tolerance_explicit(self) -> None:
        config = STEDConfig(numeric_tolerance=0.0)
        assert config.numeric_tolerance == 0.0

    def test_int_tolerance_accepted(self) -> None:
        # Plain int is fine — it's a number.
        config = STEDConfig(numeric_tolerance=1)
        assert config.numeric_tolerance == 1


class TestNumericToleranceFrozen:
    def test_numeric_tolerance_is_frozen(self) -> None:
        config = STEDConfig(numeric_tolerance=0.01)
        with pytest.raises(FrozenInstanceError):
            config.numeric_tolerance = 0.02  # type: ignore[misc]


class TestNumericToleranceValidation:
    def test_negative_tolerance_raises(self) -> None:
        with pytest.raises(ValueError, match="numeric_tolerance"):
            STEDConfig(numeric_tolerance=-0.01)

    def test_very_negative_tolerance_raises(self) -> None:
        with pytest.raises(ValueError, match="numeric_tolerance"):
            STEDConfig(numeric_tolerance=-1e9)

    def test_bool_tolerance_raises_type_error(self) -> None:
        # bool is rejected explicitly so True doesn't silently mean 1.0.
        with pytest.raises(TypeError, match="numeric_tolerance"):
            STEDConfig(numeric_tolerance=True)  # type: ignore[arg-type]

    def test_string_tolerance_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="numeric_tolerance"):
            STEDConfig(numeric_tolerance="0.01")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# max_depth configuration (F3)
# ---------------------------------------------------------------------------


class TestMaxDepthDefaults:
    def test_max_depth_default_is_none(self) -> None:
        config = STEDConfig()
        assert config.max_depth is None


class TestMaxDepthConstruction:
    def test_max_depth_one(self) -> None:
        config = STEDConfig(max_depth=1)
        assert config.max_depth == 1

    def test_max_depth_small(self) -> None:
        config = STEDConfig(max_depth=3)
        assert config.max_depth == 3

    def test_max_depth_large(self) -> None:
        config = STEDConfig(max_depth=1000)
        assert config.max_depth == 1000

    def test_max_depth_none_explicit(self) -> None:
        config = STEDConfig(max_depth=None)
        assert config.max_depth is None


class TestMaxDepthFrozen:
    def test_max_depth_is_frozen(self) -> None:
        config = STEDConfig(max_depth=5)
        with pytest.raises(FrozenInstanceError):
            config.max_depth = 7  # type: ignore[misc]


class TestMaxDepthValidation:
    def test_zero_max_depth_raises(self) -> None:
        with pytest.raises(ValueError, match="max_depth"):
            STEDConfig(max_depth=0)

    def test_negative_max_depth_raises(self) -> None:
        with pytest.raises(ValueError, match="max_depth"):
            STEDConfig(max_depth=-1)

    def test_bool_max_depth_raises_type_error(self) -> None:
        # bool subclasses int — reject explicitly.
        with pytest.raises(TypeError, match="max_depth"):
            STEDConfig(max_depth=True)  # type: ignore[arg-type]

    def test_float_max_depth_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="max_depth"):
            STEDConfig(max_depth=1.5)  # type: ignore[arg-type]

    def test_string_max_depth_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="max_depth"):
            STEDConfig(max_depth="3")  # type: ignore[arg-type]


class TestNumericToleranceAndMaxDepthCombined:
    def test_both_set_together(self) -> None:
        config = STEDConfig(numeric_tolerance=1e-9, max_depth=10)
        assert config.numeric_tolerance == pytest.approx(1e-9)
        assert config.max_depth == 10

    def test_config_with_new_fields_still_hashable(self) -> None:
        config = STEDConfig(numeric_tolerance=0.5, max_depth=4)
        assert isinstance(hash(config), int)

    def test_config_with_new_fields_equality(self) -> None:
        c1 = STEDConfig(numeric_tolerance=1e-6, max_depth=5)
        c2 = STEDConfig(numeric_tolerance=1e-6, max_depth=5)
        assert c1 == c2

    def test_config_with_new_fields_inequality(self) -> None:
        c1 = STEDConfig(numeric_tolerance=1e-6, max_depth=5)
        c2 = STEDConfig(numeric_tolerance=1e-7, max_depth=5)
        assert c1 != c2


# ---------------------------------------------------------------------------
# aliases field
# ---------------------------------------------------------------------------


class TestSTEDConfigAliasesDefaults:
    def test_aliases_default_empty_tuple(self) -> None:
        config = STEDConfig()
        assert config.aliases == ()

    def test_aliases_empty_tuple_explicit(self) -> None:
        config = STEDConfig(aliases=())
        assert config.aliases == ()


class TestSTEDConfigAliasesConstruction:
    def test_single_alias_pair_constructs(self) -> None:
        config = STEDConfig(aliases=(("uid", "user_id"),))
        assert config.aliases == (("uid", "user_id"),)

    def test_multiple_alias_pairs_constructs(self) -> None:
        config = STEDConfig(
            aliases=(("uid", "user_id"), ("addr", "address"), ("dob", "date_of_birth"))
        )
        assert len(config.aliases) == 3

    def test_aliases_frozen(self) -> None:
        config = STEDConfig(aliases=(("uid", "user_id"),))
        with pytest.raises(FrozenInstanceError):
            config.aliases = (("a", "b"),)  # type: ignore[misc]

    def test_aliases_hashable(self) -> None:
        config = STEDConfig(aliases=(("uid", "user_id"),))
        assert isinstance(hash(config), int)

    def test_aliases_equality(self) -> None:
        c1 = STEDConfig(aliases=(("uid", "user_id"),))
        c2 = STEDConfig(aliases=(("uid", "user_id"),))
        assert c1 == c2

    def test_aliases_inequality(self) -> None:
        c1 = STEDConfig(aliases=(("uid", "user_id"),))
        c2 = STEDConfig(aliases=(("uid", "other_id"),))
        assert c1 != c2


class TestSTEDConfigAliasesValidation:
    def test_non_tuple_container_raises_typeerror(self) -> None:
        with pytest.raises(TypeError, match="aliases must be a tuple"):
            STEDConfig(aliases=[("a", "b")])  # type: ignore[arg-type]

    def test_entry_not_a_tuple_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match="aliases entries must be 2-tuples"):
            STEDConfig(aliases=(["a", "b"],))  # type: ignore[arg-type]

    def test_entry_wrong_arity_one_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly 2 elements"):
            STEDConfig(aliases=(("a",),))  # type: ignore[arg-type]

    def test_entry_wrong_arity_three_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly 2 elements"):
            STEDConfig(aliases=(("a", "b", "c"),))  # type: ignore[arg-type]

    def test_entry_non_string_element_raises(self) -> None:
        with pytest.raises(ValueError, match="elements must be strings"):
            STEDConfig(aliases=(("a", 42),))  # type: ignore[arg-type]

    def test_entry_empty_left_string_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            STEDConfig(aliases=(("", "user_id"),))

    def test_entry_empty_right_string_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            STEDConfig(aliases=(("uid", ""),))


class TestSTEDConfigAliasesPreservesOtherFields:
    def test_aliases_combined_with_existing(self) -> None:
        config = STEDConfig(
            ignore_paths=("/timestamp",),
            aliases=(("uid", "user_id"),),
            null_equals_missing=True,
        )
        assert config.aliases == (("uid", "user_id"),)
        assert config.ignore_paths == ("/timestamp",)
        assert config.null_equals_missing is True

    def test_default_aliases_does_not_disturb_array_mode(self) -> None:
        config = STEDConfig()
        assert config.aliases == ()
        assert config.array_comparison_mode == ArrayComparisonMode.ORDERED
