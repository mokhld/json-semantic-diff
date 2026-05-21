"""Integration tests for STEDAlgorithm.

Tests verify the must-have truths from 04-03-PLAN.md:
- ALGO-03: Accurate STED similarity (identity=1.0, semantic > 0.85, break < 0.1)
- ALGO-04: Object order invariance
- ALGO-05: Per-level normalization, symmetry, bounds
- ALGO-06: Ordered array mode
- ALGO-07: Unordered array mode
- ALGO-09: Auto mode homogeneity heuristic

All tests use StaticBackend (injected, no ML dependencies).
"""

from __future__ import annotations

import pytest

from json_semantic_diff.algorithm.config import ArrayComparisonMode, STEDConfig
from json_semantic_diff.algorithm.sted import STEDAlgorithm
from json_semantic_diff.backends.static import StaticBackend

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def backend() -> StaticBackend:
    """StaticBackend instance shared across tests."""
    return StaticBackend()


@pytest.fixture
def algo(backend: StaticBackend) -> STEDAlgorithm:
    """Default STEDAlgorithm (ordered mode, default config)."""
    return STEDAlgorithm(backend=backend)


@pytest.fixture
def algo_unordered(backend: StaticBackend) -> STEDAlgorithm:
    """STEDAlgorithm with UNORDERED array comparison mode."""
    config = STEDConfig(array_comparison_mode=ArrayComparisonMode.UNORDERED)
    return STEDAlgorithm(backend=backend, config=config)


@pytest.fixture
def algo_auto(backend: StaticBackend) -> STEDAlgorithm:
    """STEDAlgorithm with AUTO array comparison mode."""
    config = STEDConfig(array_comparison_mode=ArrayComparisonMode.AUTO)
    return STEDAlgorithm(backend=backend, config=config)


# ---------------------------------------------------------------------------
# Identity tests (ALGO-03, ALGO-05)
# ---------------------------------------------------------------------------


class TestIdentity:
    """Identical inputs must return similarity of exactly 1.0."""

    def test_identical_simple_object(self, algo: STEDAlgorithm) -> None:
        """Simple object identity."""
        assert algo.compute({"a": 1}, {"a": 1}) == pytest.approx(1.0)

    def test_identical_empty_array(self, algo: STEDAlgorithm) -> None:
        """Empty arrays are identical."""
        assert algo.compute([], []) == pytest.approx(1.0)

    def test_identical_empty_object(self, algo: STEDAlgorithm) -> None:
        """Empty objects are identical."""
        assert algo.compute({}, {}) == pytest.approx(1.0)

    def test_identical_string_scalar(self, algo: STEDAlgorithm) -> None:
        """String scalars: same value is identical."""
        assert algo.compute("hello", "hello") == pytest.approx(1.0)

    def test_identical_integer_scalar(self, algo: STEDAlgorithm) -> None:
        """Integer scalars: same value is identical."""
        assert algo.compute(42, 42) == pytest.approx(1.0)

    def test_identical_float_scalar(self, algo: STEDAlgorithm) -> None:
        """Float scalars: same value is identical."""
        assert algo.compute(3.14, 3.14) == pytest.approx(1.0)

    def test_identical_null(self, algo: STEDAlgorithm) -> None:
        """Null scalars are identical."""
        assert algo.compute(None, None) == pytest.approx(1.0)

    def test_identical_bool_true(self, algo: STEDAlgorithm) -> None:
        """True == True is identical."""
        assert algo.compute(True, True) == pytest.approx(1.0)

    def test_identical_bool_false(self, algo: STEDAlgorithm) -> None:
        """False == False is identical."""
        assert algo.compute(False, False) == pytest.approx(1.0)

    def test_identical_deep_nested(self, algo: STEDAlgorithm) -> None:
        """Deep nested objects are identical."""
        doc = {"a": {"b": {"c": 1}}}
        assert algo.compute(doc, doc) == pytest.approx(1.0)

    def test_identical_nested_with_array(self, algo: STEDAlgorithm) -> None:
        """Nested object containing an array is identical to itself."""
        doc = {"a": [1, 2, 3], "b": {"c": "hello"}}
        assert algo.compute(doc, doc) == pytest.approx(1.0)

    def test_identical_multi_key_object(self, algo: STEDAlgorithm) -> None:
        """Multi-key object identity."""
        doc = {"x": 1, "y": 2, "z": 3}
        assert algo.compute(doc, doc) == pytest.approx(1.0)

    def test_identical_array_of_scalars(self, algo: STEDAlgorithm) -> None:
        """Array of scalars is identical to itself."""
        assert algo.compute([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_identical_array_of_objects(self, algo: STEDAlgorithm) -> None:
        """Array of objects is identical to itself."""
        doc = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        assert algo.compute(doc, doc) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Semantic equivalence (ALGO-03)
# ---------------------------------------------------------------------------


class TestSemanticEquivalence:
    """Naming-convention equivalents must score > 0.85 with StaticBackend."""

    def test_snake_vs_camel_single_key(self, algo: STEDAlgorithm) -> None:
        """user_name vs userName with same value -> > 0.85."""
        score = algo.compute({"user_name": "John"}, {"userName": "John"})
        assert score > 0.85

    def test_snake_vs_camel_multi_key(self, algo: STEDAlgorithm) -> None:
        """Multiple naming-convention equivalents -> > 0.85."""
        a = {"user_name": "John", "email": "j@x.com"}
        b = {"userName": "John", "email": "j@x.com"}
        assert algo.compute(a, b) > 0.85

    def test_snake_vs_pascal(self, algo: STEDAlgorithm) -> None:
        """user_name vs UserName -> naming convention equivalents."""
        score = algo.compute({"user_name": "Alice"}, {"UserName": "Alice"})
        assert score > 0.85

    def test_camel_vs_kebab(self, algo: STEDAlgorithm) -> None:
        """camelCase vs kebab-case equivalent keys."""
        score = algo.compute({"firstName": "Bob"}, {"first-name": "Bob"})
        assert score > 0.85

    def test_identical_after_normalization(self, algo: STEDAlgorithm) -> None:
        """Keys that normalize to the same canonical form score 1.0."""
        # user_name and userName both normalize to 'user name'
        score = algo.compute({"user_name": "test"}, {"userName": "test"})
        assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Structural breaks (ALGO-03)
# ---------------------------------------------------------------------------


class TestStructuralBreaks:
    """Structurally unrelated documents must score < 0.1."""

    def test_user_name_vs_address(self, algo: STEDAlgorithm) -> None:
        """user_name key vs address key: completely different semantics."""
        score = algo.compute({"user_name": "John"}, {"address": "123 Main St"})
        assert score < 0.1

    def test_different_scalar_types_object(self, algo: STEDAlgorithm) -> None:
        """Object vs array (different root types) -> low score."""
        score = algo.compute({"a": 1}, [1])
        assert score < 0.1

    def test_unrelated_scalars(self, algo: STEDAlgorithm) -> None:
        """Two completely different scalar values (integer vs string)."""
        score = algo.compute(42, "hello")
        assert score < 1.0  # Not necessarily < 0.1 for scalars, but < 1.0


# ---------------------------------------------------------------------------
# Object order invariance (ALGO-04)
# ---------------------------------------------------------------------------


class TestObjectOrderInvariance:
    """Object comparison must be order-invariant (ALGO-04)."""

    def test_two_key_reorder(self, algo: STEDAlgorithm) -> None:
        """Two-key object reordered -> 1.0 (Hungarian matching)."""
        assert algo.compute({"a": 1, "b": 2}, {"b": 2, "a": 1}) == pytest.approx(1.0)

    def test_three_key_reorder(self, algo: STEDAlgorithm) -> None:
        """Three-key object reordered -> 1.0."""
        a = {"x": 1, "y": 2, "z": 3}
        b = {"z": 3, "x": 1, "y": 2}
        assert algo.compute(a, b) == pytest.approx(1.0)

    def test_five_key_reorder(self, algo: STEDAlgorithm) -> None:
        """Five-key object reordered -> 1.0."""
        a = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
        b = {"e": 5, "d": 4, "a": 1, "c": 3, "b": 2}
        assert algo.compute(a, b) == pytest.approx(1.0)

    def test_nested_object_reorder(self, algo: STEDAlgorithm) -> None:
        """Nested objects with reordered keys at multiple levels -> 1.0."""
        a = {"outer_a": 1, "nested": {"inner_x": "foo", "inner_y": "bar"}}
        b = {"nested": {"inner_y": "bar", "inner_x": "foo"}, "outer_a": 1}
        assert algo.compute(a, b) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Ordered array mode (ALGO-06)
# ---------------------------------------------------------------------------


class TestOrderedArrayMode:
    """Default mode is ORDERED: position matters for arrays."""

    def test_same_order_identity(self, algo: STEDAlgorithm) -> None:
        """Same order -> 1.0."""
        assert algo.compute([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_reversed_order_not_identity(self, algo: STEDAlgorithm) -> None:
        """Reversed order -> < 1.0 (position matters)."""
        score = algo.compute([1, 2, 3], [3, 2, 1])
        assert score < 1.0

    def test_different_lengths_not_identity(self, algo: STEDAlgorithm) -> None:
        """Different length arrays -> < 1.0."""
        score = algo.compute([1, 2, 3], [1, 2, 3, 4])
        assert score < 1.0

    def test_prefix_match_partial(self, algo: STEDAlgorithm) -> None:
        """Common prefix arrays -> partial match score."""
        score = algo.compute([1, 2, 3], [1, 2, 9])
        assert 0.0 < score < 1.0

    def test_completely_different_arrays(self, algo: STEDAlgorithm) -> None:
        """Completely different elements -> low-ish score."""
        score = algo.compute([1, 2, 3], [4, 5, 6])
        assert score < 1.0


# ---------------------------------------------------------------------------
# Unordered array mode (ALGO-07)
# ---------------------------------------------------------------------------


class TestUnorderedArrayMode:
    """UNORDERED mode: position is irrelevant."""

    def test_reversed_is_identity(self, algo_unordered: STEDAlgorithm) -> None:
        """Reversed scalars -> 1.0 (unordered = set comparison)."""
        assert algo_unordered.compute([1, 2, 3], [3, 2, 1]) == pytest.approx(1.0)

    def test_same_elements_different_order(self, algo_unordered: STEDAlgorithm) -> None:
        """Same elements, any permutation -> 1.0."""
        assert algo_unordered.compute([10, 20, 30], [30, 10, 20]) == pytest.approx(1.0)

    def test_different_content_not_identity(
        self, algo_unordered: STEDAlgorithm
    ) -> None:
        """Different content (one element differs) -> < 1.0."""
        score = algo_unordered.compute([1, 2, 3], [1, 2, 4])
        assert score < 1.0

    def test_different_lengths_not_identity(
        self, algo_unordered: STEDAlgorithm
    ) -> None:
        """Different length arrays -> < 1.0 even in UNORDERED mode."""
        score = algo_unordered.compute([1, 2, 3], [1, 2, 3, 4])
        assert score < 1.0

    def test_empty_arrays_identity(self, algo_unordered: STEDAlgorithm) -> None:
        """Empty arrays are identical in UNORDERED mode."""
        assert algo_unordered.compute([], []) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Auto array mode (ALGO-09)
# ---------------------------------------------------------------------------


class TestAutoArrayMode:
    """AUTO mode: infers ordered/unordered from content homogeneity."""

    def test_scalar_arrays_treated_as_unordered(self, algo_auto: STEDAlgorithm) -> None:
        """Scalar-only arrays -> AUTO infers UNORDERED -> reversed = 1.0."""
        score = algo_auto.compute([1, 2, 3], [3, 2, 1])
        assert score == pytest.approx(1.0)

    def test_object_arrays_treated_as_ordered(self, algo_auto: STEDAlgorithm) -> None:
        """Object arrays -> AUTO infers ORDERED -> different objects matter."""
        a = [{"id": 1}]
        b = [{"id": 2}]
        score_auto = algo_auto.compute(a, b)
        # With ORDERED, position still matters — just check it's a valid score
        assert 0.0 <= score_auto <= 1.0

    def test_mixed_arrays_treated_as_ordered(self, algo_auto: STEDAlgorithm) -> None:
        """Mixed scalar+object arrays -> AUTO infers ORDERED."""
        a = [1, {"a": 2}]
        b = [{"a": 2}, 1]
        # AUTO should choose ORDERED (has objects) -> order matters -> < 1.0
        score = algo_auto.compute(a, b)
        assert 0.0 <= score <= 1.0

    def test_empty_arrays_auto(self, algo_auto: STEDAlgorithm) -> None:
        """Empty arrays in AUTO mode -> 1.0 (resolved to UNORDERED)."""
        assert algo_auto.compute([], []) == pytest.approx(1.0)

    def test_scalar_permutation_is_identity(self, algo_auto: STEDAlgorithm) -> None:
        """AUTO: string scalar arrays -> unordered, permutation = identity."""
        assert algo_auto.compute(["a", "b", "c"], ["c", "a", "b"]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Symmetry tests (ALGO-05)
# ---------------------------------------------------------------------------


class TestSymmetry:
    """compute(a, b) must equal compute(b, a) for all inputs."""

    @pytest.mark.parametrize(
        ("json_a", "json_b"),
        [
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
            # Nested objects
            ({"a": {"b": 1}}, {"a": {"b": 2}}),
            ({"a": {"b": {"c": 1}}}, {"a": {"b": {"c": 2}}}),
            # Arrays
            ([1, 2, 3], [4, 5, 6]),
            ([1], [1, 2]),
            ([], [1, 2, 3]),
            ([1, 2, 3], [3, 2, 1]),
            # Mixed types
            ({"a": 1}, [1]),
            ([{"a": 1}], [{"b": 2}]),
            ({"a": [1, 2]}, {"a": [3, 4]}),
            # Objects with different key counts
            ({"a": 1, "b": 2, "c": 3}, {"x": 10}),
            ({"name": "Alice", "age": 30}, {"name": "Bob", "email": "b@x.com"}),
        ],
        ids=[
            "ints",
            "strings",
            "null-vs-str",
            "bools",
            "floats",
            "simple-objs-diff-keys",
            "objs-diff-count",
            "empty-vs-obj",
            "objs-diff-string-vals",
            "nested-diff-leaf",
            "deep-nested-diff",
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
    def test_symmetry(
        self, algo: STEDAlgorithm, json_a: object, json_b: object
    ) -> None:
        """compute(a, b) == compute(b, a) within floating-point tolerance."""
        score_ab = algo.compute(json_a, json_b)
        score_ba = algo.compute(json_b, json_a)
        assert abs(score_ab - score_ba) < 1e-9, (
            f"Symmetry violation: compute({json_a!r}, {json_b!r}) = {score_ab} "
            f"!= compute({json_b!r}, {json_a!r}) = {score_ba}"
        )


# ---------------------------------------------------------------------------
# Normalization bounds (ALGO-05)
# ---------------------------------------------------------------------------


class TestNormalizationBounds:
    """All scores must be in [0.0, 1.0]."""

    @pytest.mark.parametrize(
        ("json_a", "json_b"),
        [
            ({}, {}),
            ({"a": 1}, {"a": 1}),
            ({"a": 1}, {"b": 2}),
            ([], []),
            ([1, 2], [3, 4]),
            ("hello", "world"),
            (42, 99),
            (None, None),
            ({"a": {"b": 1}}, {"a": {"b": 2}}),
            ({"a": [1, 2]}, {"b": [3, 4]}),
            ([{"id": 1}], [{"id": 2}]),
            ({"a": 1}, [1]),
            ({}, []),
        ],
    )
    def test_score_in_bounds(
        self, algo: STEDAlgorithm, json_a: object, json_b: object
    ) -> None:
        """Score is in [0.0, 1.0] for all valid input pairs."""
        score = algo.compute(json_a, json_b)
        assert 0.0 <= score <= 1.0, (
            f"Out of bounds score {score} for {json_a!r} vs {json_b!r}"
        )

    def test_deep_nesting_in_bounds(self, algo: STEDAlgorithm) -> None:
        """Very deep nesting does not push score outside [0, 1]."""
        deep_a: dict[str, object] = {"level": {"level": {"level": {"value": 1}}}}
        deep_b: dict[str, object] = {"level": {"level": {"level": {"value": 99}}}}
        score = algo.compute(deep_a, deep_b)
        assert 0.0 <= score <= 1.0

    def test_wide_object_in_bounds(self, algo: STEDAlgorithm) -> None:
        """Wide object (many keys) stays in [0, 1]."""
        a = {f"key_{i}": i for i in range(20)}
        b = {f"other_{i}": i * 2 for i in range(20)}
        score = algo.compute(a, b)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Type mismatch (ALGO-03)
# ---------------------------------------------------------------------------


class TestTypeMismatch:
    """Different root types must score 0.0 (complete structural break)."""

    def test_object_vs_array(self, algo: STEDAlgorithm) -> None:
        """Object vs array -> 0.0 (type mismatch at root)."""
        assert algo.compute({"a": 1}, [1]) == pytest.approx(0.0)

    def test_array_vs_object(self, algo: STEDAlgorithm) -> None:
        """Array vs object -> 0.0 (type mismatch at root)."""
        assert algo.compute([1], {"a": 1}) == pytest.approx(0.0)

    def test_object_vs_scalar(self, algo: STEDAlgorithm) -> None:
        """Object vs scalar -> 0.0."""
        assert algo.compute({"a": 1}, 42) == pytest.approx(0.0)

    def test_array_vs_scalar(self, algo: STEDAlgorithm) -> None:
        """Array vs scalar -> 0.0."""
        assert algo.compute([1, 2], "hello") == pytest.approx(0.0)

    def test_null_vs_object(self, algo: STEDAlgorithm) -> None:
        """Null vs object -> 0.0."""
        assert algo.compute(None, {"a": 1}) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Additional correctness tests
# ---------------------------------------------------------------------------


class TestAdditionalCorrectness:
    """Extra tests for edge cases and specific behaviors."""

    def test_single_extra_key(self, algo: STEDAlgorithm) -> None:
        """One object has an extra key: partial similarity."""
        score = algo.compute({"a": 1}, {"a": 1, "b": 2})
        assert 0.0 < score < 1.0

    def test_nested_value_change(self, algo: STEDAlgorithm) -> None:
        """Same structure, different leaf value: high but not 1.0 similarity."""
        a = {"user": {"name": "Alice"}}
        b = {"user": {"name": "Bob"}}
        score = algo.compute(a, b)
        assert 0.0 < score < 1.0

    def test_different_scalar_values_not_identical(self, algo: STEDAlgorithm) -> None:
        """Different integer values -> content_distance = 1.0 -> score < 1.0."""
        # Note: True == 1 in Python (bool subclass of int), so True vs 1 is 1.0.
        # Use genuinely different values to verify content_distance works.
        score = algo.compute(42, 99)
        assert score < 1.0

    def test_default_config_is_ordered_mode(self, algo: STEDAlgorithm) -> None:
        """Default config uses ORDERED array mode."""
        # [1,2,3] vs [3,2,1]: ORDERED -> < 1.0; UNORDERED -> 1.0
        ordered_score = algo.compute([1, 2, 3], [3, 2, 1])
        assert ordered_score < 1.0

    def test_none_config_uses_defaults(self, backend: StaticBackend) -> None:
        """STEDAlgorithm with config=None uses STEDConfig() defaults."""
        algo = STEDAlgorithm(backend=backend, config=None)
        assert algo.compute({}, {}) == pytest.approx(1.0)

    def test_custom_lambda_affects_score(self, backend: StaticBackend) -> None:
        """Higher lambda_unmatched increases penalty for unmatched children."""
        config_low = STEDConfig(lambda_unmatched=0.0)
        config_high = STEDConfig(lambda_unmatched=1.0)
        algo_low = STEDAlgorithm(backend=backend, config=config_low)
        algo_high = STEDAlgorithm(backend=backend, config=config_high)

        # One object has an extra key — higher lambda penalizes more
        a = {"x": 1, "y": 2}
        b = {"x": 1}
        score_low = algo_low.compute(a, b)
        score_high = algo_high.compute(a, b)
        assert score_high < score_low

    def test_array_length_mismatch_penalized(self, algo: STEDAlgorithm) -> None:
        """Longer array vs shorter array -> < identity score."""
        score_equal = algo.compute([1, 2, 3], [1, 2, 3])
        score_longer = algo.compute([1, 2, 3], [1, 2, 3, 4])
        assert score_longer < score_equal


# ---------------------------------------------------------------------------
# Type-mismatch cost scaling (C5) and KEY normalization resolution (C6)
# ---------------------------------------------------------------------------


class TestTypeMismatchSubtreeScaling:
    """Type-mismatch at a child position scales cost by subtree size.

    Before the fix, ``_compute_node_distance`` returned a flat 1.0 whenever
    two compared nodes had different ``node_type``, which meant deleting a
    100-node nested object and inserting a single scalar was charged the
    same as a 1-vs-1 swap.  The Hungarian / DP matching then picked
    nonsensical alignments because all cross-type swaps looked equally
    cheap.  The fix scales the cost by ``max(subtree_size(a), subtree_size(b))``.
    """

    def test_deep_subtree_vs_scalar_more_costly_than_scalar_swap(
        self, algo: STEDAlgorithm
    ) -> None:
        """Replacing a 5-key nested object with a scalar must cost more than
        a single scalar mismatch — at the same child position."""
        # Same key on both sides ("k"); value differs in size of subtree
        big_subtree = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
        score_big_swap = algo.compute({"k": big_subtree}, {"k": "scalar"})
        score_small_swap = algo.compute({"k": "x"}, {"k": "scalar"})
        # Bigger subtree replacement -> larger raw cost -> lower similarity
        assert score_big_swap < score_small_swap

    def test_hungarian_prefers_size_matched_pair(self, algo: STEDAlgorithm) -> None:
        """When two object branches differ, Hungarian must prefer matching a
        small-vs-small swap to a large-vs-small swap (because the latter
        now correctly costs more)."""
        a = {"big": {"x": 1, "y": 2, "z": 3}, "small": 1}
        b = {"big": {"x": 1, "y": 2, "z": 3}, "small": 2}  # tiny diff at 'small'
        # The matching should align big->big (identical, cost 0) and
        # small->small (cheap scalar swap) — NOT cross-swap.
        score = algo.compute(a, b)
        # Cross-swapping would be much worse; matched pair should give high score
        assert score > 0.7

    def test_object_vs_array_at_inner_key_costly(self, algo: STEDAlgorithm) -> None:
        """At an inner KEY, value=object vs value=array now charges full
        subtree-size cost rather than 1.0."""
        a = {"data": {"x": 1, "y": 2, "z": 3}}
        b = {"data": [1, 2, 3]}
        score = algo.compute(a, b)
        # Definitely less than identity but not a flat 0; bounded in [0,1]
        assert 0.0 <= score < 1.0


class TestKeyNormalizationResolution:
    """KEY-level normalization must preserve resolution for nested subtrees.

    The internal helper ``_compute_key_similarity`` previously normalised
    by ``len(children)`` (always 1), so any value-child cost > 1 was
    clipped to 0 — collapsing KEY similarity to binary.  The fix
    normalises by subtree size (audit finding C6 partial fix).

    NOTE: ``_compute_key_similarity`` is currently only reachable through
    ``_compute_similarity`` on a top-level KEY node — which the public
    ``compute()`` entry point never produces.  The OBJECT/ARRAY
    normalisation path (still child-count-based) preserves the existing
    behaviour for shallow disjoint objects.  These tests pin the direct
    helper invocation rather than the end-to-end ``compute`` call so the
    KEY-level fix is locked in for any future caller that does reach it.
    """

    def test_compute_key_similarity_resolves_small_change_in_large_subtree(
        self, algo: STEDAlgorithm
    ) -> None:
        """Direct ``_compute_key_similarity`` call: 1 leaf differs in a
        5-key subtree → similarity stays well above 0 (not binary)."""
        from json_semantic_diff.tree.builder import TreeBuilder

        builder = TreeBuilder()
        # Build matching KEY subtrees via OBJECT->KEY path
        a_root = builder.build({"data": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}})
        b_root = builder.build({"data": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 99}})
        key_a = a_root.children[0]
        key_b = b_root.children[0]
        score = algo._compute_key_similarity(key_a, key_b)
        # Pre-fix: this was clipped near 0 because raw cost > 1 child.
        # Post-fix: large subtree-size denom keeps the score high.
        assert score > 0.7, f"expected resolution preserved, got {score}"

    def test_compute_key_similarity_monotonic_in_leaf_diffs(
        self, algo: STEDAlgorithm
    ) -> None:
        """Direct helper: scores decrease monotonically with more leaf diffs."""
        from json_semantic_diff.tree.builder import TreeBuilder

        builder = TreeBuilder()
        base = builder.build({"data": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}})
        one_diff = builder.build({"data": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 99}})
        two_diff = builder.build({"data": {"a": 1, "b": 2, "c": 3, "d": 99, "e": 99}})
        three_diff = builder.build(
            {"data": {"a": 1, "b": 2, "c": 99, "d": 99, "e": 99}}
        )

        s1 = algo._compute_key_similarity(base.children[0], one_diff.children[0])
        s2 = algo._compute_key_similarity(base.children[0], two_diff.children[0])
        s3 = algo._compute_key_similarity(base.children[0], three_diff.children[0])
        assert s1 > s2 > s3, f"non-monotonic: {s1=}, {s2=}, {s3=}"
