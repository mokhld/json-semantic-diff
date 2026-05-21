"""Tests for CostFunctions: cost_insert, cost_delete, cost_update.

Covers:
- cost_insert always returns 1.0 (unit cost)
- cost_delete always returns 1.0 (unit cost)
- cost_update for identical nodes returns 0.0
- cost_update blends structural similarity (w_s) and content distance (w_c)
- cost_update is symmetric: cost_update(a, b) == cost_update(b, a)
- cost_update handles KEY, SCALAR, OBJECT, ARRAY, ELEMENT node types
- cost_update with different NodeTypes gives structural penalty
- cost_update with same type but different KEY labels gives blended cost
- cost_update with SCALAR nodes respects value equality
"""

from __future__ import annotations

import pytest

from json_semantic_diff.algorithm.config import STEDConfig
from json_semantic_diff.algorithm.costs import cost_delete, cost_insert, cost_update
from json_semantic_diff.backends.static import StaticBackend
from json_semantic_diff.tree.nodes import NodeType, TreeNode

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def backend() -> StaticBackend:
    return StaticBackend()


@pytest.fixture
def default_config() -> STEDConfig:
    return STEDConfig()


@pytest.fixture
def struct_heavy_config() -> STEDConfig:
    """Config that heavily weights structural similarity (w_s=0.8, w_c=0.2)."""
    return STEDConfig(w_s=0.8, w_c=0.2)


@pytest.fixture
def content_heavy_config() -> STEDConfig:
    """Config that heavily weights content distance (w_s=0.2, w_c=0.8)."""
    return STEDConfig(w_s=0.2, w_c=0.8)


def make_key_node(label: str, path: str = "/key") -> TreeNode:
    """Create a KEY TreeNode with the given label."""
    return TreeNode(node_type=NodeType.KEY, label=label, path=path, raw_label=label)


def make_scalar_node(value: object, path: str = "/val") -> TreeNode:
    """Create a SCALAR TreeNode with the given value."""
    return TreeNode(
        node_type=NodeType.SCALAR,
        label=str(value),
        path=path,
        value=value,
    )


def make_object_node(path: str = "/obj") -> TreeNode:
    return TreeNode(node_type=NodeType.OBJECT, label="", path=path)


def make_array_node(path: str = "/arr") -> TreeNode:
    return TreeNode(node_type=NodeType.ARRAY, label="", path=path)


def make_element_node(path: str = "/elem") -> TreeNode:
    return TreeNode(node_type=NodeType.ELEMENT, label="", path=path)


# ---------------------------------------------------------------------------
# cost_insert
# ---------------------------------------------------------------------------


class TestCostInsert:
    def test_key_node_returns_one(self) -> None:
        node = make_key_node("user_name")
        assert cost_insert(node) == 1.0

    def test_scalar_node_returns_one(self) -> None:
        node = make_scalar_node(42)
        assert cost_insert(node) == 1.0

    def test_object_node_returns_one(self) -> None:
        node = make_object_node()
        assert cost_insert(node) == 1.0

    def test_array_node_returns_one(self) -> None:
        node = make_array_node()
        assert cost_insert(node) == 1.0

    def test_element_node_returns_one(self) -> None:
        node = make_element_node()
        assert cost_insert(node) == 1.0

    def test_returns_float(self) -> None:
        node = make_key_node("x")
        result = cost_insert(node)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# cost_delete
# ---------------------------------------------------------------------------


class TestCostDelete:
    def test_key_node_returns_one(self) -> None:
        node = make_key_node("user_name")
        assert cost_delete(node) == 1.0

    def test_scalar_node_returns_one(self) -> None:
        node = make_scalar_node("hello")
        assert cost_delete(node) == 1.0

    def test_object_node_returns_one(self) -> None:
        node = make_object_node()
        assert cost_delete(node) == 1.0

    def test_array_node_returns_one(self) -> None:
        node = make_array_node()
        assert cost_delete(node) == 1.0

    def test_element_node_returns_one(self) -> None:
        node = make_element_node()
        assert cost_delete(node) == 1.0

    def test_returns_float(self) -> None:
        node = make_key_node("x")
        result = cost_delete(node)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# cost_update — identical nodes
# ---------------------------------------------------------------------------


class TestCostUpdateIdentical:
    def test_identical_key_nodes_returns_zero(
        self, backend: StaticBackend, default_config: STEDConfig
    ) -> None:
        a = make_key_node("user_name")
        b = make_key_node("user_name")
        assert cost_update(a, b, backend, default_config) == pytest.approx(0.0)

    def test_identical_scalar_string_returns_zero(
        self, backend: StaticBackend, default_config: STEDConfig
    ) -> None:
        a = make_scalar_node("hello")
        b = make_scalar_node("hello")
        assert cost_update(a, b, backend, default_config) == pytest.approx(0.0)

    def test_identical_scalar_int_returns_zero(
        self, backend: StaticBackend, default_config: STEDConfig
    ) -> None:
        a = make_scalar_node(42)
        b = make_scalar_node(42)
        assert cost_update(a, b, backend, default_config) == pytest.approx(0.0)

    def test_identical_scalar_none_returns_zero(
        self, backend: StaticBackend, default_config: STEDConfig
    ) -> None:
        a = make_scalar_node(None)
        b = make_scalar_node(None)
        assert cost_update(a, b, backend, default_config) == pytest.approx(0.0)

    def test_identical_object_nodes_returns_zero(
        self, backend: StaticBackend, default_config: STEDConfig
    ) -> None:
        a = make_object_node()
        b = make_object_node()
        assert cost_update(a, b, backend, default_config) == pytest.approx(0.0)

    def test_identical_array_nodes_returns_zero(
        self, backend: StaticBackend, default_config: STEDConfig
    ) -> None:
        a = make_array_node()
        b = make_array_node()
        assert cost_update(a, b, backend, default_config) == pytest.approx(0.0)

    def test_identical_element_nodes_returns_zero(
        self, backend: StaticBackend, default_config: STEDConfig
    ) -> None:
        a = make_element_node()
        b = make_element_node()
        assert cost_update(a, b, backend, default_config) == pytest.approx(0.0)

    def test_normalized_equivalent_key_nodes_returns_zero(
        self, backend: StaticBackend, default_config: STEDConfig
    ) -> None:
        # "user_name" and "userName" normalize to the same tokens
        # StaticBackend.similarity("user_name", "userName") == 1.0
        a = make_key_node("user_name")
        b = make_key_node("userName")
        result = cost_update(a, b, backend, default_config)
        # gamma_struct = 1.0 - 1.0 = 0.0; gamma_content = 0.0; result = 0.0
        assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# cost_update — KEY node label differences
# ---------------------------------------------------------------------------


class TestCostUpdateKeyNodes:
    def test_totally_different_key_labels(
        self, backend: StaticBackend, default_config: STEDConfig
    ) -> None:
        a = make_key_node("username")
        b = make_key_node("address")
        result = cost_update(a, b, backend, default_config)
        # Should be positive since labels differ
        assert result > 0.0
        assert result <= 1.0

    def test_key_cost_is_in_unit_range(
        self, backend: StaticBackend, default_config: STEDConfig
    ) -> None:
        a = make_key_node("first_name")
        b = make_key_node("last_name")
        result = cost_update(a, b, backend, default_config)
        assert 0.0 <= result <= 1.0

    def test_key_cost_with_struct_heavy_config_higher_for_different_labels(
        self,
        backend: StaticBackend,
        struct_heavy_config: STEDConfig,
        content_heavy_config: STEDConfig,
    ) -> None:
        # Both labels differ — structural component dominates in struct_heavy
        a = make_key_node("alpha")
        b = make_key_node("omega")
        struct_cost = cost_update(a, b, backend, struct_heavy_config)
        content_cost = cost_update(a, b, backend, content_heavy_config)
        # KEY nodes have no content distance, so content_heavy gives lower cost
        # gamma_content = 0 for KEY nodes (they are not SCALAR)
        # struct_heavy = 0.8 * gamma_struct + 0.2 * 0 = 0.8 * gamma_struct
        # content_heavy = 0.2 * gamma_struct + 0.8 * 0 = 0.2 * gamma_struct
        assert struct_cost >= content_cost


# ---------------------------------------------------------------------------
# cost_update — SCALAR node value differences
# ---------------------------------------------------------------------------


class TestCostUpdateScalarNodes:
    def test_different_scalar_values_gives_content_penalty(
        self, backend: StaticBackend, default_config: STEDConfig
    ) -> None:
        a = make_scalar_node("hello")
        b = make_scalar_node("world")
        result = cost_update(a, b, backend, default_config)
        # gamma_struct = 0 (same type), gamma_content = 1 (different values)
        # cost = 0.5 * 0 + 0.5 * 1 = 0.5
        assert result == pytest.approx(0.5)

    def test_different_scalar_ints_gives_content_penalty(
        self, backend: StaticBackend, default_config: STEDConfig
    ) -> None:
        a = make_scalar_node(1)
        b = make_scalar_node(99)
        result = cost_update(a, b, backend, default_config)
        # Same type SCALAR, different values -> gamma_content = 1.0
        assert result == pytest.approx(0.5)

    def test_scalar_content_heavy_config(
        self, backend: StaticBackend, content_heavy_config: STEDConfig
    ) -> None:
        a = make_scalar_node("foo")
        b = make_scalar_node("bar")
        result = cost_update(a, b, backend, content_heavy_config)
        # gamma_struct = 0 (same type), gamma_content = 1 (different values)
        # cost = 0.2 * 0 + 0.8 * 1 = 0.8
        assert result == pytest.approx(0.8)

    def test_scalar_struct_heavy_config(
        self, backend: StaticBackend, struct_heavy_config: STEDConfig
    ) -> None:
        a = make_scalar_node("foo")
        b = make_scalar_node("bar")
        result = cost_update(a, b, backend, struct_heavy_config)
        # gamma_struct = 0 (same type), gamma_content = 1 (different values)
        # cost = 0.8 * 0 + 0.2 * 1 = 0.2
        assert result == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# cost_update — different NodeType (structural penalty)
# ---------------------------------------------------------------------------


class TestCostUpdateDifferentTypes:
    def test_object_vs_array_incurs_full_structural_cost(
        self, backend: StaticBackend, default_config: STEDConfig
    ) -> None:
        a = make_object_node()
        b = make_array_node()
        result = cost_update(a, b, backend, default_config)
        # gamma_struct = 1.0 (different types), gamma_content = 0.0 (non-scalar)
        # cost = 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        assert result == pytest.approx(0.5)

    def test_object_vs_element_incurs_structural_cost(
        self, backend: StaticBackend, default_config: STEDConfig
    ) -> None:
        a = make_object_node()
        b = make_element_node()
        result = cost_update(a, b, backend, default_config)
        # gamma_struct = 1.0, gamma_content = 0.0
        assert result == pytest.approx(0.5)

    def test_array_vs_element_incurs_structural_cost(
        self, backend: StaticBackend, default_config: STEDConfig
    ) -> None:
        a = make_array_node()
        b = make_element_node()
        result = cost_update(a, b, backend, default_config)
        assert result == pytest.approx(0.5)

    def test_key_vs_scalar_has_structural_penalty(
        self, backend: StaticBackend, default_config: STEDConfig
    ) -> None:
        # KEY node: gamma_struct from backend similarity
        # SCALAR node: non-KEY, gamma_struct = 0 if same type, 1 if different
        # KEY vs SCALAR => different types
        a = make_key_node("username")
        b = make_scalar_node("hello")
        result = cost_update(a, b, backend, default_config)
        # KEY is computed differently from non-KEY
        # For a KEY node: gamma_struct = 1 - backend.similarity(a.label, b.label)
        # For b (SCALAR/non-KEY): gamma_struct = 1.0 (different types)
        # This tests that cross-type comparisons still yield a result
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# cost_update — symmetry
# ---------------------------------------------------------------------------


class TestCostUpdateSymmetry:
    def test_key_nodes_symmetric(
        self, backend: StaticBackend, default_config: STEDConfig
    ) -> None:
        a = make_key_node("username")
        b = make_key_node("address")
        assert cost_update(a, b, backend, default_config) == pytest.approx(
            cost_update(b, a, backend, default_config)
        )

    def test_scalar_nodes_symmetric(
        self, backend: StaticBackend, default_config: STEDConfig
    ) -> None:
        a = make_scalar_node("hello")
        b = make_scalar_node("world")
        assert cost_update(a, b, backend, default_config) == pytest.approx(
            cost_update(b, a, backend, default_config)
        )

    def test_object_vs_array_symmetric(
        self, backend: StaticBackend, default_config: STEDConfig
    ) -> None:
        a = make_object_node()
        b = make_array_node()
        assert cost_update(a, b, backend, default_config) == pytest.approx(
            cost_update(b, a, backend, default_config)
        )

    def test_symmetry_with_struct_heavy_config(
        self, backend: StaticBackend, struct_heavy_config: STEDConfig
    ) -> None:
        a = make_key_node("first_name")
        b = make_key_node("last_name")
        assert cost_update(a, b, backend, struct_heavy_config) == pytest.approx(
            cost_update(b, a, backend, struct_heavy_config)
        )

    def test_symmetry_mixed_types(
        self, backend: StaticBackend, default_config: STEDConfig
    ) -> None:
        a = make_object_node()
        b = make_element_node()
        assert cost_update(a, b, backend, default_config) == pytest.approx(
            cost_update(b, a, backend, default_config)
        )

    def test_symmetry_different_scalars(
        self, backend: StaticBackend, default_config: STEDConfig
    ) -> None:
        a = make_scalar_node(True)
        b = make_scalar_node(False)
        assert cost_update(a, b, backend, default_config) == pytest.approx(
            cost_update(b, a, backend, default_config)
        )


# ---------------------------------------------------------------------------
# cost_update — result range
# ---------------------------------------------------------------------------


class TestCostUpdateRange:
    def test_result_is_in_unit_interval(
        self, backend: StaticBackend, default_config: STEDConfig
    ) -> None:
        pairs = [
            (make_key_node("x"), make_key_node("y")),
            (make_scalar_node(1), make_scalar_node(2)),
            (make_object_node(), make_array_node()),
            (make_element_node(), make_object_node()),
            (make_key_node("a"), make_scalar_node("b")),
        ]
        for node_a, node_b in pairs:
            result = cost_update(node_a, node_b, backend, default_config)
            assert 0.0 <= result <= 1.0, (
                f"Out of range: {node_a.node_type} vs {node_b.node_type} -> {result}"
            )

    def test_returns_float(
        self, backend: StaticBackend, default_config: STEDConfig
    ) -> None:
        a = make_key_node("a")
        b = make_key_node("b")
        result = cost_update(a, b, backend, default_config)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# cost_update — bool vs int distinction (H2 / M7)
# ---------------------------------------------------------------------------


class TestCostUpdateBoolVsInt:
    """Bool MUST NOT collapse to int 0/1.  Policy: bool is its own type."""

    def test_true_vs_int_one_is_distance_one(
        self, backend: StaticBackend, default_config: STEDConfig
    ) -> None:
        """True and 1 are distinct SCALAR types — content_distance must be 1.0.

        Bug guard for H2: ``True == 1`` is True in Python, so a naive ``==``
        check would collapse the content distance to 0.0.  We must keep
        bool and int as separate domains.
        """
        a = make_scalar_node(True)
        b = make_scalar_node(1)
        # gamma_struct = 0 (both SCALAR), gamma_content must be 1.0
        # cost = 0.5 * 0 + 0.5 * 1.0 = 0.5
        result = cost_update(a, b, backend, default_config)
        assert result == pytest.approx(0.5)

    def test_false_vs_int_zero_is_distance_one(
        self, backend: StaticBackend, default_config: STEDConfig
    ) -> None:
        a = make_scalar_node(False)
        b = make_scalar_node(0)
        result = cost_update(a, b, backend, default_config)
        assert result == pytest.approx(0.5)

    def test_true_vs_true_is_zero(
        self, backend: StaticBackend, default_config: STEDConfig
    ) -> None:
        a = make_scalar_node(True)
        b = make_scalar_node(True)
        assert cost_update(a, b, backend, default_config) == pytest.approx(0.0)

    def test_false_vs_true_is_distance_one(
        self, backend: StaticBackend, default_config: STEDConfig
    ) -> None:
        a = make_scalar_node(False)
        b = make_scalar_node(True)
        assert cost_update(a, b, backend, default_config) == pytest.approx(0.5)

    def test_true_vs_string_one_not_coerced(self, backend: StaticBackend) -> None:
        """Even with type_coercion=True, bool is NOT coerced into '1' / '0'."""
        config = STEDConfig(type_coercion=True)
        a = make_scalar_node(True)
        b = make_scalar_node("1")
        result = cost_update(a, b, backend, config)
        # Bool excludes coercion; structural=0 (both SCALAR), content=1
        assert result == pytest.approx(0.5)

    def test_true_vs_float_one_not_coerced(self, backend: StaticBackend) -> None:
        config = STEDConfig(type_coercion=True)
        a = make_scalar_node(True)
        b = make_scalar_node(1.0)
        # Bool excludes coercion -> content distance 1.0
        result = cost_update(a, b, backend, config)
        assert result == pytest.approx(0.5)

    def test_int_vs_float_same_value_zero_distance(
        self, backend: StaticBackend, default_config: STEDConfig
    ) -> None:
        """5 == 5.0 (numeric equivalence, no type-coercion needed)."""
        a = make_scalar_node(5)
        b = make_scalar_node(5.0)
        # int/float compatible -> distance 0
        result = cost_update(a, b, backend, default_config)
        assert result == pytest.approx(0.0)

    def test_string_vs_int_coercion_path_still_works(
        self, backend: StaticBackend
    ) -> None:
        """type_coercion=True still coerces '5' <-> 5 (M7 policy preserved)."""
        config = STEDConfig(type_coercion=True)
        a = make_scalar_node("5")
        b = make_scalar_node(5)
        result = cost_update(a, b, backend, config)
        # Coerced via float() — both equal 5.0 — content distance 0
        assert result == pytest.approx(0.0)

    def test_string_vs_int_no_coercion_default(
        self, backend: StaticBackend, default_config: STEDConfig
    ) -> None:
        """Without type_coercion, '5' and 5 are different."""
        a = make_scalar_node("5")
        b = make_scalar_node(5)
        result = cost_update(a, b, backend, default_config)
        assert result == pytest.approx(0.5)
