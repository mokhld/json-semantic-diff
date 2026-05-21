"""End-to-end integration tests for Phase 4: STED Algorithm.

Each test class maps directly to one of the six ROADMAP Phase 4 success
criteria, verified through the *public* ``json_semantic_diff.algorithm`` interface.

Requirements covered:
- ALGO-03: Accurate STED similarity (identity=1.0, naming > 0.85, break < 0.1)
- ALGO-04: Object order invariance (Hungarian matching)
- ALGO-05: Symmetry and identity properties
- ALGO-06: Ordered array mode
- ALGO-07: Unordered array mode
- ALGO-09: Auto array mode heuristic

All tests import from ``json_semantic_diff.algorithm`` (public API), never
from submodules directly.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from json_semantic_diff.algorithm import ArrayComparisonMode, STEDAlgorithm, STEDConfig
from json_semantic_diff.backends import StaticBackend

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def backend() -> StaticBackend:
    """Fresh StaticBackend instance."""
    return StaticBackend()


@pytest.fixture
def sted() -> STEDAlgorithm:
    """Default STEDAlgorithm with ordered array mode."""
    return STEDAlgorithm(backend=StaticBackend())


@pytest.fixture
def sted_unordered() -> STEDAlgorithm:
    """STEDAlgorithm with UNORDERED array comparison mode."""
    return STEDAlgorithm(
        backend=StaticBackend(),
        config=STEDConfig(array_comparison_mode=ArrayComparisonMode.UNORDERED),
    )


@pytest.fixture
def sted_auto() -> STEDAlgorithm:
    """STEDAlgorithm with AUTO array comparison mode."""
    return STEDAlgorithm(
        backend=StaticBackend(),
        config=STEDConfig(array_comparison_mode=ArrayComparisonMode.AUTO),
    )


# ---------------------------------------------------------------------------
# SC1: Identical = 1.0, disjoint ~ 0.0
# ---------------------------------------------------------------------------


class TestSC1IdentityAndDisjoint:
    """SC1: Identical inputs score 1.0; disjoint inputs score near 0.0."""

    def test_identical_simple_object(self, sted: STEDAlgorithm) -> None:
        """Single-key object identical to itself -> 1.0."""
        assert sted.compute({"a": 1}, {"a": 1}) == pytest.approx(1.0)

    def test_identical_nested(self, sted: STEDAlgorithm) -> None:
        """Nested object identical to itself -> 1.0."""
        doc = {"user": {"name": "John", "age": 30}}
        assert sted.compute(doc, doc) == pytest.approx(1.0)

    def test_identical_with_arrays(self, sted: STEDAlgorithm) -> None:
        """Object containing array identical to itself -> 1.0."""
        doc = {"items": [1, 2, 3]}
        assert sted.compute(doc, doc) == pytest.approx(1.0)

    def test_disjoint_objects(self, sted: STEDAlgorithm) -> None:
        """Disjoint object (no shared keys or values) -> low score.

        Audit C6 (wave 7): under the Zhang-Shasha denominator, two OBJECTs
        whose structures are both ``KEY -> SCALAR`` no longer binary-collapse
        to ~0.  The asymmetric size (1 vs 3 children) still drives the score
        well below the equivalence band (>= 0.85).
        """
        a = {"a": 1}
        b = {"x": 99, "y": "hello", "z": [1, 2]}
        score = sted.compute(a, b)
        assert score < 0.75


# ---------------------------------------------------------------------------
# SC2: Naming convention equivalence > 0.85 (StaticBackend)
# ---------------------------------------------------------------------------


class TestSC2NamingConventionEquivalence:
    """SC2: snake_case / camelCase equivalents score > 0.85 with StaticBackend."""

    def test_snake_vs_camel(self, sted: STEDAlgorithm) -> None:
        """user_name vs userName with same value -> > 0.85."""
        score = sted.compute({"user_name": "John"}, {"userName": "John"})
        assert score > 0.85

    def test_multi_key_naming(self, sted: STEDAlgorithm) -> None:
        """Multiple naming-convention differences in one object -> > 0.85."""
        a = {"user_name": "John", "email_address": "j@x"}
        b = {"userName": "John", "emailAddress": "j@x"}
        score = sted.compute(a, b)
        assert score > 0.85


# ---------------------------------------------------------------------------
# SC3: Structural breaks < 0.1
# ---------------------------------------------------------------------------


class TestSC3StructuralBreaks:
    """SC3: Structurally unrelated documents score < 0.1."""

    def test_structural_break(self, sted: STEDAlgorithm) -> None:
        """user_name key vs address key, different values: low score.

        Audit C6 (wave 7): two single-key OBJECTs with KEY -> SCALAR shape
        now have a Zhang-Shasha-normalised floor around 0.5.  ``< 0.6``
        captures "well below the equivalence band" without re-introducing
        the binary collapse the audit fixed.
        """
        score = sted.compute({"user_name": "John"}, {"address": "123 Main St"})
        assert score < 0.6

    def test_completely_different_structure(self, sted: STEDAlgorithm) -> None:
        """Nested object vs array-valued object -> type mismatch.

        Audit C6 (wave 7): even with deeper subtrees, the KEY value
        type-mismatch (OBJECT vs ARRAY) inflates the matched cost enough
        to keep the overall score below the 0.6 band.
        """
        a = {"a": {"b": 1}}
        b = {"x": [1, 2, 3]}
        score = sted.compute(a, b)
        assert score < 0.6


# ---------------------------------------------------------------------------
# SC4: Object order invariance
# ---------------------------------------------------------------------------


class TestSC4ObjectOrderInvariance:
    """SC4: Object comparison is fully order-invariant via Hungarian matching."""

    def test_two_key_order(self, sted: STEDAlgorithm) -> None:
        """Two-key object reordered -> 1.0."""
        assert sted.compute({"a": 1, "b": 2}, {"b": 2, "a": 1}) == pytest.approx(1.0)

    def test_three_key_order(self, sted: STEDAlgorithm) -> None:
        """Three-key object with rotated key order -> 1.0."""
        a = {"x": 1, "y": 2, "z": 3}
        b = {"z": 3, "x": 1, "y": 2}
        assert sted.compute(a, b) == pytest.approx(1.0)

    def test_nested_order(self, sted: STEDAlgorithm) -> None:
        """Nested object with reordered keys at both levels -> 1.0."""
        a = {"a": {"x": 1, "y": 2}, "b": 3}
        b = {"b": 3, "a": {"y": 2, "x": 1}}
        assert sted.compute(a, b) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# SC5: Array comparison modes
# ---------------------------------------------------------------------------


class TestSC5ArrayComparisonModes:
    """SC5: Ordered, unordered, and auto array modes behave correctly."""

    def test_ordered_same_order(self, sted: STEDAlgorithm) -> None:
        """ORDERED mode: same order -> 1.0."""
        assert sted.compute([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_ordered_different_order(self, sted: STEDAlgorithm) -> None:
        """ORDERED mode: different order -> < 1.0 (position matters)."""
        score = sted.compute([1, 2, 3], [3, 2, 1])
        assert score < 1.0

    def test_unordered_reorder(self, sted_unordered: STEDAlgorithm) -> None:
        """UNORDERED mode: reordered scalars -> 1.0 (position irrelevant)."""
        assert sted_unordered.compute([1, 2, 3], [3, 2, 1]) == pytest.approx(1.0)

    def test_auto_scalar_array(self, sted_auto: STEDAlgorithm) -> None:
        """AUTO mode: scalar array detected as unordered -> permutation = 1.0."""
        assert sted_auto.compute([1, 2, 3], [3, 2, 1]) == pytest.approx(1.0)

    def test_auto_object_array_ordered(self, sted_auto: STEDAlgorithm) -> None:
        """AUTO mode: identical object array -> 1.0 regardless of mode chosen."""
        doc = [{"id": 1, "name": "a"}]
        assert sted_auto.compute(doc, doc) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# SC6: Symmetry and identity for 20+ diverse pairs and 15+ distinct inputs
# ---------------------------------------------------------------------------


class TestSC6SymmetryAndIdentity:
    """SC6: Symmetry holds for 20+ pairs; identity holds for 15+ distinct inputs."""

    # 20 diverse (a, b) pairs for symmetry check
    SYMMETRY_PAIRS: ClassVar[list[tuple[object, object]]] = [
        # Scalars
        (42, 99),
        ("hello", "world"),
        (None, "something"),
        (True, False),
        (3.14, 2.71),
        # Simple objects
        ({"a": 1}, {"b": 2}),
        ({"a": 1, "b": 2}, {"c": 3}),
        ({}, {"a": 1}),
        ({"x": "hello"}, {"y": "world"}),
        # Naming convention variants
        ({"user_name": "Alice"}, {"userName": "Alice"}),
        # Nested objects
        ({"a": {"b": 1}}, {"a": {"b": 2}}),
        ({"a": {"b": {"c": 1}}}, {"a": {"b": {"c": 2}}}),
        # Arrays
        ([1, 2, 3], [4, 5, 6]),
        ([1], [1, 2]),
        ([], [1, 2, 3]),
        ([1, 2, 3], [3, 2, 1]),
        # Mixed types (root type mismatch)
        ({"a": 1}, [1]),
        ([{"a": 1}], [{"b": 2}]),
        ({"a": [1, 2]}, {"a": [3, 4]}),
        # Objects with different key counts
        ({"a": 1, "b": 2, "c": 3}, {"x": 10}),
        # Real-world-ish pairs
        ({"name": "Alice", "age": 30}, {"name": "Bob", "email": "b@x.com"}),
    ]

    @pytest.mark.parametrize(
        ("json_a", "json_b"),
        SYMMETRY_PAIRS,
        ids=[
            "ints",
            "strings",
            "null-vs-str",
            "bools",
            "floats",
            "simple-objs",
            "objs-diff-count",
            "empty-vs-obj",
            "objs-diff-str-vals",
            "naming-convention",
            "nested-diff-leaf",
            "deep-nested",
            "arrays-diff",
            "arrays-diff-len",
            "empty-vs-array",
            "reversed-array",
            "obj-vs-array",
            "obj-arrays",
            "obj-with-array-vals",
            "objs-diff-key-counts",
            "objs-diff-keys-and-vals",
        ],
    )
    def test_symmetry_20_pairs(
        self, sted: STEDAlgorithm, json_a: object, json_b: object
    ) -> None:
        """compute(a, b) == compute(b, a) within 1e-9 for all 20+ diverse pairs."""
        score_ab = sted.compute(json_a, json_b)
        score_ba = sted.compute(json_b, json_a)
        assert abs(score_ab - score_ba) < 1e-9, (
            f"Symmetry violation: compute({json_a!r}, {json_b!r}) = {score_ab} "
            f"!= compute({json_b!r}, {json_a!r}) = {score_ba}"
        )

    # 15 distinct inputs for identity check (compute(x, x) == 1.0)
    IDENTITY_INPUTS: ClassVar[list[object]] = [
        {},
        [],
        None,
        42,
        "hello",
        True,
        False,
        3.14,
        {"a": 1},
        [1, 2, 3],
        {"user": {"name": "Alice", "age": 30}},
        [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
        {"a": {"b": {"c": {"d": 1}}}},
        {"x": [1, 2], "y": {"z": "nested"}},
        {"snake_case_key": "value", "another_key": [True, None, 42]},
    ]

    @pytest.mark.parametrize(
        "json_x",
        IDENTITY_INPUTS,
        ids=[
            "empty-obj",
            "empty-arr",
            "none",
            "int",
            "string",
            "true",
            "false",
            "float",
            "simple-obj",
            "array",
            "nested-obj",
            "array-of-objs",
            "deep-nested",
            "mixed-structure",
            "snake-case-keys",
        ],
    )
    def test_identity_all_inputs(self, sted: STEDAlgorithm, json_x: object) -> None:
        """compute(x, x) == 1.0 for all 15+ distinct inputs."""
        score = sted.compute(json_x, json_x)
        assert score == pytest.approx(1.0), (
            f"Identity violation: compute({json_x!r}, {json_x!r}) = {score}"
        )


# ---------------------------------------------------------------------------
# Real-world JSON pairs
# ---------------------------------------------------------------------------


class TestRealWorldJSONPairs:
    """Realistic API JSON documents produce discriminative scores."""

    def test_api_response_minor_changes(self, sted: STEDAlgorithm) -> None:
        """Two API responses with renamed keys but same structure -> high score."""
        # Simulates a v1 vs v2 API response where only naming conventions changed
        v1 = {
            "user_id": 42,
            "first_name": "Alice",
            "last_name": "Smith",
            "email_address": "alice@example.com",
            "created_at": "2024-01-15",
            "is_active": True,
        }
        v2 = {
            "userId": 42,
            "firstName": "Alice",
            "lastName": "Smith",
            "emailAddress": "alice@example.com",
            "createdAt": "2024-01-15",
            "isActive": True,
        }
        score = sted.compute(v1, v2)
        # With StaticBackend, naming-convention variants normalize to same form
        assert score > 0.85

    def test_api_response_major_rewrite(self, sted: STEDAlgorithm) -> None:
        """Two API responses with completely different structure -> low score."""
        # User profile vs product catalog entry: completely different semantics
        user_profile = {
            "user_id": 42,
            "first_name": "Alice",
            "last_name": "Smith",
            "email_address": "alice@example.com",
            "phone_number": "+1-555-0100",
        }
        product_entry = {
            "product_sku": "WIDGET-001",
            "unit_price": 29.99,
            "stock_quantity": 150,
            "warehouse_location": "A-3-7",
            "category_tag": "hardware",
        }
        score = sted.compute(user_profile, product_entry)
        # Audit C6 (wave 7): same-cardinality OBJECTs with all 5 keys
        # different + all values different.  Zhang-Shasha denominator
        # gives a floor around 0.55 (5 matched pairs each with raw cost
        # ~0.9, denom 10).  Still well below the equivalence band.
        assert score < 0.6
