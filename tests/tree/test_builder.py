"""Comprehensive tests for TreeBuilder.

Covers all JSON types, nesting levels, JSON Pointer paths, key normalization,
round-trip determinism, bool/int dispatch ordering, type preservation,
empty containers, mixed arrays, and TypeError on invalid input.
"""

from __future__ import annotations

from typing import Any

import pytest

from json_semantic_diff.tree.builder import TreeBuilder
from json_semantic_diff.tree.nodes import NodeType, TreeNode

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def builder() -> TreeBuilder:
    """A fresh TreeBuilder instance for each test."""
    return TreeBuilder()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compare_structure(a: TreeNode, b: TreeNode) -> bool:
    """Recursively compare two trees for structural equality.

    Checks node_type, label, path, raw_label, and value at every node.
    Does NOT compare children by identity — recurses into them.
    """
    if (
        a.node_type != b.node_type
        or a.label != b.label
        or a.path != b.path
        or a.raw_label != b.raw_label
        or a.value != b.value
        or type(a.value) is not type(b.value)
        or len(a.children) != len(b.children)
    ):
        return False
    return all(
        _compare_structure(ac, bc)
        for ac, bc in zip(a.children, b.children, strict=True)
    )


# ---------------------------------------------------------------------------
# ROADMAP Success Criterion 1:
# {"user_name": "John"} -> Object -> Key("user name", raw="user_name") -> Scalar("John")
# ---------------------------------------------------------------------------


class TestRoadmapCriterion1:
    """SC1: Snake-case key produces normalized KEY with raw_label preserved."""

    def test_root_is_object(self, builder: TreeBuilder) -> None:
        tree = builder.build({"user_name": "John"})
        assert tree.node_type == NodeType.OBJECT

    def test_object_has_one_child(self, builder: TreeBuilder) -> None:
        tree = builder.build({"user_name": "John"})
        assert len(tree.children) == 1

    def test_key_node_label_normalized(self, builder: TreeBuilder) -> None:
        tree = builder.build({"user_name": "John"})
        key_node = tree.children[0]
        assert key_node.node_type == NodeType.KEY
        assert key_node.label == "user name"

    def test_key_node_raw_label_preserved(self, builder: TreeBuilder) -> None:
        tree = builder.build({"user_name": "John"})
        key_node = tree.children[0]
        assert key_node.raw_label == "user_name"

    def test_key_node_path(self, builder: TreeBuilder) -> None:
        tree = builder.build({"user_name": "John"})
        key_node = tree.children[0]
        assert key_node.path == "/user_name"

    def test_scalar_child_label(self, builder: TreeBuilder) -> None:
        tree = builder.build({"user_name": "John"})
        scalar_node = tree.children[0].children[0]
        assert scalar_node.node_type == NodeType.SCALAR
        assert scalar_node.label == "John"

    def test_scalar_child_value(self, builder: TreeBuilder) -> None:
        tree = builder.build({"user_name": "John"})
        scalar_node = tree.children[0].children[0]
        assert scalar_node.value == "John"
        assert isinstance(scalar_node.value, str)

    def test_scalar_child_path(self, builder: TreeBuilder) -> None:
        tree = builder.build({"user_name": "John"})
        scalar_node = tree.children[0].children[0]
        assert scalar_node.path == "/user_name"


# ---------------------------------------------------------------------------
# ROADMAP Success Criterion 2: Nested JSON with correct parent/path structure
# ---------------------------------------------------------------------------


class TestRoadmapCriterion2Nested:
    """SC2: Nested JSON converts to correctly parented tree with accurate paths."""

    def test_nested_object_structure(self, builder: TreeBuilder) -> None:
        tree = builder.build({"user": {"name": "John", "age": 30}})
        # root OBJECT -> KEY("user") -> OBJECT -> [KEY("name"), KEY("age")]
        assert tree.node_type == NodeType.OBJECT
        user_key = tree.children[0]
        assert user_key.node_type == NodeType.KEY
        assert user_key.label == "user"
        assert user_key.path == "/user"
        inner_obj = user_key.children[0]
        assert inner_obj.node_type == NodeType.OBJECT
        assert len(inner_obj.children) == 2

    def test_nested_object_paths(self, builder: TreeBuilder) -> None:
        tree = builder.build({"user": {"name": "John", "age": 30}})
        inner_obj = tree.children[0].children[0]
        paths = {child.path for child in inner_obj.children}
        assert paths == {"/user/name", "/user/age"}

    def test_nested_object_scalar_name(self, builder: TreeBuilder) -> None:
        tree = builder.build({"user": {"name": "John", "age": 30}})
        inner_obj = tree.children[0].children[0]
        name_key = next(c for c in inner_obj.children if c.raw_label == "name")
        assert name_key.children[0].node_type == NodeType.SCALAR
        assert name_key.children[0].label == "John"

    def test_nested_object_scalar_age(self, builder: TreeBuilder) -> None:
        tree = builder.build({"user": {"name": "John", "age": 30}})
        inner_obj = tree.children[0].children[0]
        age_key = next(c for c in inner_obj.children if c.raw_label == "age")
        assert age_key.children[0].label == "30"
        assert age_key.children[0].value == 30

    def test_array_of_objects_root(self, builder: TreeBuilder) -> None:
        tree = builder.build([{"a": 1}, {"b": 2}])
        assert tree.node_type == NodeType.ARRAY
        assert len(tree.children) == 2

    def test_array_of_objects_element_0(self, builder: TreeBuilder) -> None:
        tree = builder.build([{"a": 1}, {"b": 2}])
        elem0 = tree.children[0]
        assert elem0.node_type == NodeType.ELEMENT
        assert elem0.label == "0"
        assert elem0.path == "/0"
        inner = elem0.children[0]
        assert inner.node_type == NodeType.OBJECT
        assert inner.children[0].raw_label == "a"

    def test_array_of_objects_element_1_path(self, builder: TreeBuilder) -> None:
        tree = builder.build([{"a": 1}, {"b": 2}])
        elem1 = tree.children[1]
        assert elem1.path == "/1"
        key_b = elem1.children[0].children[0]
        assert key_b.path == "/1/b"

    def test_array_of_objects_scalar_value(self, builder: TreeBuilder) -> None:
        tree = builder.build([{"a": 1}, {"b": 2}])
        scalar_1 = tree.children[0].children[0].children[0].children[0]
        assert scalar_1.node_type == NodeType.SCALAR
        assert scalar_1.value == 1
        assert isinstance(scalar_1.value, int)


# ---------------------------------------------------------------------------
# ROADMAP Success Criterion 3: Key normalization via KeyNormalizer
# ---------------------------------------------------------------------------


class TestRoadmapCriterion3KeyNormalization:
    """SC3: All four naming conventions normalize via TreeBuilder."""

    def test_snake_case_normalizes(self, builder: TreeBuilder) -> None:
        tree = builder.build({"user_name": "x"})
        assert tree.children[0].label == "user name"

    def test_camel_case_normalizes(self, builder: TreeBuilder) -> None:
        tree = builder.build({"userName": "x"})
        assert tree.children[0].label == "user name"

    def test_pascal_case_normalizes(self, builder: TreeBuilder) -> None:
        tree = builder.build({"UserName": "x"})
        assert tree.children[0].label == "user name"

    def test_kebab_case_normalizes(self, builder: TreeBuilder) -> None:
        tree = builder.build({"user-name": "x"})
        assert tree.children[0].label == "user name"


# ---------------------------------------------------------------------------
# ROADMAP Success Criterion 4: Round-trip determinism
# ---------------------------------------------------------------------------


class TestRoadmapCriterion4Determinism:
    """SC4: Same input produces identical tree structure on repeat calls."""

    def test_simple_determinism(self, builder: TreeBuilder) -> None:
        data = {"a": 1}
        tree1 = builder.build(data)
        tree2 = builder.build(data)
        assert _compare_structure(tree1, tree2)

    def test_nested_determinism(self, builder: TreeBuilder) -> None:
        data = {"user": {"name": "Alice", "scores": [10, 20, 30]}}
        tree1 = builder.build(data)
        tree2 = builder.build(data)
        assert _compare_structure(tree1, tree2)

    def test_different_instances_determinism(self) -> None:
        data = {"x": [1, 2], "y": {"z": True}}
        tree1 = TreeBuilder().build(data)
        tree2 = TreeBuilder().build(data)
        assert _compare_structure(tree1, tree2)


# ---------------------------------------------------------------------------
# Scalar type dispatch tests
# ---------------------------------------------------------------------------


class TestScalarTypes:
    """All seven JSON leaf types dispatch correctly with label and value."""

    def test_string_scalar(self, builder: TreeBuilder) -> None:
        node = builder.build("hello")
        assert node.node_type == NodeType.SCALAR
        assert node.label == "hello"
        assert node.value == "hello"
        assert isinstance(node.value, str)

    def test_int_scalar(self, builder: TreeBuilder) -> None:
        node = builder.build(42)
        assert node.node_type == NodeType.SCALAR
        assert node.label == "42"
        assert node.value == 42
        assert type(node.value) is int

    def test_float_scalar(self, builder: TreeBuilder) -> None:
        node = builder.build(3.14)
        assert node.node_type == NodeType.SCALAR
        assert node.label == "3.14"
        assert node.value == pytest.approx(3.14)
        assert isinstance(node.value, float)

    def test_true_scalar(self, builder: TreeBuilder) -> None:
        node = builder.build(True)
        assert node.node_type == NodeType.SCALAR
        assert node.label == "true"
        assert node.value is True
        assert type(node.value) is bool

    def test_false_scalar(self, builder: TreeBuilder) -> None:
        node = builder.build(False)
        assert node.node_type == NodeType.SCALAR
        assert node.label == "false"
        assert node.value is False
        assert type(node.value) is bool

    def test_none_scalar(self, builder: TreeBuilder) -> None:
        node = builder.build(None)
        assert node.node_type == NodeType.SCALAR
        assert node.label == "null"
        assert node.value is None

    def test_string_vs_int_same_label_different_type(
        self, builder: TreeBuilder
    ) -> None:
        """LOCKED: "5" (str) and 5 (int) produce same label but distinct value types."""
        str_node = builder.build("5")
        int_node = builder.build(5)
        assert str_node.label == int_node.label == "5"
        assert type(str_node.value) is str
        assert type(int_node.value) is int
        assert type(str_node.value) is not type(int_node.value)


# ---------------------------------------------------------------------------
# Bool before int dispatch — regression guard
# ---------------------------------------------------------------------------


class TestBoolBeforeInt:
    """bool MUST be dispatched before int to prevent True -> "1" / False -> "0"."""

    def test_true_label_is_true_not_1(self, builder: TreeBuilder) -> None:
        node = builder.build(True)
        assert node.label == "true"
        assert node.label != "1"

    def test_false_label_is_false_not_0(self, builder: TreeBuilder) -> None:
        node = builder.build(False)
        assert node.label == "false"
        assert node.label != "0"

    def test_true_value_is_bool_not_int(self, builder: TreeBuilder) -> None:
        node = builder.build(True)
        assert type(node.value) is bool
        assert type(node.value) is not int

    def test_false_value_is_bool_not_int(self, builder: TreeBuilder) -> None:
        node = builder.build(False)
        assert type(node.value) is bool
        assert type(node.value) is not int


# ---------------------------------------------------------------------------
# Array tests
# ---------------------------------------------------------------------------


class TestArrayDispatch:
    """Array nodes have ELEMENT children with index labels and correct paths."""

    def test_array_root(self, builder: TreeBuilder) -> None:
        node = builder.build([1, 2, 3])
        assert node.node_type == NodeType.ARRAY

    def test_array_child_count(self, builder: TreeBuilder) -> None:
        node = builder.build([1, 2, 3])
        assert len(node.children) == 3

    def test_array_element_types(self, builder: TreeBuilder) -> None:
        node = builder.build([1, 2, 3])
        for child in node.children:
            assert child.node_type == NodeType.ELEMENT

    def test_array_element_labels(self, builder: TreeBuilder) -> None:
        node = builder.build([1, 2, 3])
        labels = [child.label for child in node.children]
        assert labels == ["0", "1", "2"]

    def test_array_element_paths(self, builder: TreeBuilder) -> None:
        node = builder.build([1, 2, 3])
        paths = [child.path for child in node.children]
        assert paths == ["/0", "/1", "/2"]

    def test_array_scalar_values(self, builder: TreeBuilder) -> None:
        node = builder.build([1, 2, 3])
        for i, elem in enumerate(node.children, start=1):
            scalar = elem.children[0]
            assert scalar.node_type == NodeType.SCALAR
            assert scalar.value == i


# ---------------------------------------------------------------------------
# Empty container tests
# ---------------------------------------------------------------------------


class TestEmptyContainers:
    """Empty dict and list produce valid OBJECT/ARRAY nodes with zero children."""

    def test_empty_dict(self, builder: TreeBuilder) -> None:
        node = builder.build({})
        assert node.node_type == NodeType.OBJECT
        assert len(node.children) == 0

    def test_empty_list(self, builder: TreeBuilder) -> None:
        node = builder.build([])
        assert node.node_type == NodeType.ARRAY
        assert len(node.children) == 0

    def test_empty_dict_path(self, builder: TreeBuilder) -> None:
        node = builder.build({})
        assert node.path == ""

    def test_empty_list_path(self, builder: TreeBuilder) -> None:
        node = builder.build([])
        assert node.path == ""


# ---------------------------------------------------------------------------
# Deeply nested test
# ---------------------------------------------------------------------------


class TestDeeplyNested:
    """Deeply nested structures produce correct paths at every level."""

    def test_three_levels_deep(self, builder: TreeBuilder) -> None:
        tree = builder.build({"a": {"b": {"c": "deep"}}})
        # root OBJECT -> KEY("a") -> OBJECT -> KEY("b") -> OBJECT -> KEY("c") -> SCALAR("deep")
        key_a = tree.children[0]
        assert key_a.label == "a"
        assert key_a.path == "/a"

        obj_b_level = key_a.children[0]
        key_b = obj_b_level.children[0]
        assert key_b.label == "b"
        assert key_b.path == "/a/b"

        obj_c_level = key_b.children[0]
        key_c = obj_c_level.children[0]
        assert key_c.label == "c"
        assert key_c.path == "/a/b/c"

        scalar = key_c.children[0]
        assert scalar.node_type == NodeType.SCALAR
        assert scalar.label == "deep"
        assert scalar.path == "/a/b/c"


# ---------------------------------------------------------------------------
# Mixed type array
# ---------------------------------------------------------------------------


class TestMixedTypeArray:
    """Arrays with elements of different JSON types all dispatch correctly."""

    def test_mixed_array_element_count(self, builder: TreeBuilder) -> None:
        node = builder.build([1, "two", True, None, {"key": "val"}])
        assert len(node.children) == 5

    def test_mixed_array_int_element(self, builder: TreeBuilder) -> None:
        node = builder.build([1, "two", True, None, {"key": "val"}])
        scalar = node.children[0].children[0]
        assert scalar.node_type == NodeType.SCALAR
        assert scalar.value == 1
        assert type(scalar.value) is int

    def test_mixed_array_string_element(self, builder: TreeBuilder) -> None:
        node = builder.build([1, "two", True, None, {"key": "val"}])
        scalar = node.children[1].children[0]
        assert scalar.label == "two"
        assert isinstance(scalar.value, str)

    def test_mixed_array_bool_element(self, builder: TreeBuilder) -> None:
        node = builder.build([1, "two", True, None, {"key": "val"}])
        scalar = node.children[2].children[0]
        assert scalar.label == "true"
        assert type(scalar.value) is bool

    def test_mixed_array_none_element(self, builder: TreeBuilder) -> None:
        node = builder.build([1, "two", True, None, {"key": "val"}])
        scalar = node.children[3].children[0]
        assert scalar.label == "null"
        assert scalar.value is None

    def test_mixed_array_object_element(self, builder: TreeBuilder) -> None:
        node = builder.build([1, "two", True, None, {"key": "val"}])
        inner_obj = node.children[4].children[0]
        assert inner_obj.node_type == NodeType.OBJECT
        assert inner_obj.children[0].label == "key"


# ---------------------------------------------------------------------------
# TypeError on invalid input
# ---------------------------------------------------------------------------


class TestTypeError:
    """Non-JSON types raise TypeError."""

    def test_set_raises_type_error(self, builder: TreeBuilder) -> None:
        with pytest.raises(TypeError, match="Unsupported JSON value type"):
            builder.build(set())  # type: ignore[arg-type]

    def test_tuple_raises_type_error(self, builder: TreeBuilder) -> None:
        with pytest.raises(TypeError, match="Unsupported JSON value type"):
            builder.build((1, 2))  # type: ignore[arg-type]

    def test_bytes_raises_type_error(self, builder: TreeBuilder) -> None:
        with pytest.raises(TypeError, match="Unsupported JSON value type"):
            builder.build(b"hello")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Cycle detection (M11): self-referencing containers raise ValueError
# ---------------------------------------------------------------------------


class TestCycleDetection:
    """Cyclic Python structures must raise ValueError, not recurse forever."""

    def test_dict_self_reference(self, builder: TreeBuilder) -> None:
        a: dict[str, Any] = {}
        a["self"] = a
        with pytest.raises(ValueError, match="Cyclic structure detected"):
            builder.build(a)

    def test_list_self_reference(self, builder: TreeBuilder) -> None:
        lst: list[Any] = []
        lst.append(lst)
        with pytest.raises(ValueError, match="Cyclic structure detected"):
            builder.build(lst)

    def test_dict_indirect_cycle(self, builder: TreeBuilder) -> None:
        """a -> b -> a forms a two-step cycle."""
        a: dict[str, Any] = {}
        b: dict[str, Any] = {"back": a}
        a["forward"] = b
        with pytest.raises(ValueError, match="Cyclic structure detected"):
            builder.build(a)

    def test_list_dict_mixed_cycle(self, builder: TreeBuilder) -> None:
        a: dict[str, Any] = {}
        lst: list[Any] = [a]
        a["lst"] = lst
        with pytest.raises(ValueError, match="Cyclic structure detected"):
            builder.build(a)

    def test_repeated_non_cyclic_subtree_does_not_raise(
        self, builder: TreeBuilder
    ) -> None:
        """A DAG that re-uses the same subtree twice is NOT a cycle —
        both references appear at different points but neither is on the
        active recursion stack when the other is processed."""
        shared = {"x": 1}
        doc = {"a": shared, "b": shared}
        # Should not raise; both branches build the same subtree shape
        tree = builder.build(doc)
        assert tree.node_type == NodeType.OBJECT
        assert len(tree.children) == 2


# ---------------------------------------------------------------------------
# Non-string keys (M12): JSON spec requires string keys
# ---------------------------------------------------------------------------


class TestNonStringKeys:
    """Python dicts allow any hashable key, but JSON requires strings.

    Reject explicitly with TypeError rather than silently corrupting paths
    or crashing inside the regex normalizer.
    """

    def test_int_key_raises_type_error(self, builder: TreeBuilder) -> None:
        with pytest.raises(TypeError, match="JSON object keys must be strings"):
            builder.build({1: "value"})

    def test_float_key_raises_type_error(self, builder: TreeBuilder) -> None:
        with pytest.raises(TypeError, match="JSON object keys must be strings"):
            builder.build({1.5: "value"})

    def test_tuple_key_raises_type_error(self, builder: TreeBuilder) -> None:
        with pytest.raises(TypeError, match="JSON object keys must be strings"):
            builder.build({(1, 2): "value"})

    def test_none_key_raises_type_error(self, builder: TreeBuilder) -> None:
        with pytest.raises(TypeError, match="JSON object keys must be strings"):
            builder.build({None: "value"})  # type: ignore[dict-item]

    def test_bool_key_raises_type_error(self, builder: TreeBuilder) -> None:
        """``True`` is hashable and dict-compatible but not a JSON string key."""
        with pytest.raises(TypeError, match="JSON object keys must be strings"):
            builder.build({True: "value"})  # type: ignore[dict-item]

    def test_nested_int_key_reports_inner_path(self, builder: TreeBuilder) -> None:
        """Nested non-string key surfaces with the parent path in the error."""
        with pytest.raises(TypeError, match="JSON object keys must be strings"):
            builder.build({"outer": {2: "v"}})


# ---------------------------------------------------------------------------
# Root path propagation
# ---------------------------------------------------------------------------


class TestPathPropagation:
    """JSON Pointer paths are correctly built at every depth level."""

    def test_root_object_path(self, builder: TreeBuilder) -> None:
        tree = builder.build({"x": 1})
        assert tree.path == ""

    def test_root_array_path(self, builder: TreeBuilder) -> None:
        tree = builder.build([1])
        assert tree.path == ""

    def test_root_scalar_path(self, builder: TreeBuilder) -> None:
        tree = builder.build("hello")
        assert tree.path == ""

    def test_nested_array_element_path(self, builder: TreeBuilder) -> None:
        tree = builder.build({"items": [10, 20]})
        items_key = tree.children[0]
        inner_array = items_key.children[0]
        assert inner_array.path == "/items"
        assert inner_array.children[0].path == "/items/0"
        assert inner_array.children[1].path == "/items/1"

    def test_scalar_inside_nested_object_path(self, builder: TreeBuilder) -> None:
        tree = builder.build({"a": {"b": 42}})
        scalar = tree.children[0].children[0].children[0].children[0]
        assert scalar.path == "/a/b"


# ---------------------------------------------------------------------------
# JSON Pointer (RFC 6901) escaping of key path segments
# ---------------------------------------------------------------------------


class TestJsonPointerEscaping:
    """Object key path segments must be RFC 6901-escaped.

    Per RFC 6901 section 4:
        ``~`` → ``~0``
        ``/`` → ``~1``
    with the ``~`` substitution applied first so a literal ``~`` in a key
    does not corrupt the ``~`` produced by the ``/`` substitution.
    """

    def test_slash_in_key_is_escaped(self, builder: TreeBuilder) -> None:
        tree = builder.build({"a/b": 1})
        key_node = tree.children[0]
        assert key_node.path == "/a~1b"
        # raw_label preserves the original key untouched
        assert key_node.raw_label == "a/b"

    def test_tilde_in_key_is_escaped(self, builder: TreeBuilder) -> None:
        tree = builder.build({"a~b": 1})
        key_node = tree.children[0]
        assert key_node.path == "/a~0b"
        assert key_node.raw_label == "a~b"

    def test_both_tilde_and_slash_in_key_tilde_first(
        self, builder: TreeBuilder
    ) -> None:
        """Order matters: ``~`` first so the literal ``~`` does not eat ``~1``.

        Input key ``"~/"`` must become ``~0~1`` (tilde then slash), NOT
        ``~01`` (which would be the result of escaping slash first then
        tilde — wrong per RFC 6901).
        """
        tree = builder.build({"~/": 1})
        key_node = tree.children[0]
        assert key_node.path == "/~0~1"
        assert key_node.raw_label == "~/"

    def test_round_trip_nested_escaping(self, builder: TreeBuilder) -> None:
        """Nested keys containing ``/`` and ``~`` both get escaped at every depth."""
        tree = builder.build({"a/b": {"c~d": 1}})
        outer_key = tree.children[0]
        assert outer_key.path == "/a~1b"
        inner_obj = outer_key.children[0]
        inner_key = inner_obj.children[0]
        assert inner_key.path == "/a~1b/c~0d"
        scalar = inner_key.children[0]
        assert scalar.path == "/a~1b/c~0d"

    def test_slash_only_key_at_root(self, builder: TreeBuilder) -> None:
        tree = builder.build({"/": "v"})
        key_node = tree.children[0]
        assert key_node.path == "/~1"

    def test_tilde_only_key_at_root(self, builder: TreeBuilder) -> None:
        tree = builder.build({"~": "v"})
        key_node = tree.children[0]
        assert key_node.path == "/~0"

    def test_array_index_path_unchanged(self, builder: TreeBuilder) -> None:
        """Array indices are decimal integers — no escaping needed."""
        tree = builder.build({"a/b": [10, 20]})
        outer_key = tree.children[0]
        inner_array = outer_key.children[0]
        assert inner_array.children[0].path == "/a~1b/0"
        assert inner_array.children[1].path == "/a~1b/1"

    def test_ascii_key_path_is_byte_identical(self, builder: TreeBuilder) -> None:
        """Keys with no ``~`` or ``/`` produce unchanged paths (no double-encoding)."""
        tree = builder.build({"plain_key": 1})
        assert tree.children[0].path == "/plain_key"
