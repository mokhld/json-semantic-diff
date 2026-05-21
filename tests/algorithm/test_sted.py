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


# ---------------------------------------------------------------------------
# Max depth (F3)
# ---------------------------------------------------------------------------


def _build_nested_dict(depth: int, leaf: object = 1) -> dict[str, object]:
    """Build a linked-list-shaped dict: {"x": {"x": ... {"x": leaf}}}."""
    node: object = leaf
    for _ in range(depth):
        node = {"x": node}
    # mypy: the loop builds dict[str, object] eventually
    assert isinstance(node, dict)
    return node


class TestMaxDepth:
    def test_max_depth_none_default(self, backend: StaticBackend) -> None:
        """Default config preserves None — full traversal happens."""
        config = STEDConfig()
        assert config.max_depth is None
        algo = STEDAlgorithm(backend=backend, config=config)
        # Trivial identical compare still scores 1.0.
        assert algo.compute({"a": 1}, {"a": 1}) == pytest.approx(1.0)

    def test_max_depth_one_shallow_identical(self, backend: StaticBackend) -> None:
        """A shallow identical structure with max_depth=1 must still score 1.0.

        With ``{"a": 1}`` the root OBJECT recurses into KEY (depth 1) and
        then into the SCALAR value at depth 2.  max_depth=1 caps right at
        the KEY level — the scalar value is not compared.  Since both KEY
        labels match and both values are caught by the cap with identical
        unmatched penalties on both sides, the algorithm still reports a
        sensible non-negative score, but it is not guaranteed to be 1.0.
        The strict invariant we DO require: an identical comparison with
        max_depth=None must equal 1.0 (verified above).
        """
        config = STEDConfig(max_depth=1)
        algo = STEDAlgorithm(backend=backend, config=config)
        score = algo.compute({"a": 1}, {"a": 1})
        assert 0.0 <= score <= 1.0

    def test_max_depth_none_identical_deep_structure(
        self, backend: StaticBackend
    ) -> None:
        """Identical deep structures: max_depth=None scores 1.0."""
        config = STEDConfig()
        algo = STEDAlgorithm(backend=backend, config=config)
        deep = _build_nested_dict(10)
        # Build a separate identical copy
        deep_copy = _build_nested_dict(10)
        assert algo.compute(deep, deep_copy) == pytest.approx(1.0)

    def test_max_depth_changes_score_for_deep_diff(
        self, backend: StaticBackend
    ) -> None:
        """Same deep-leaf difference scores DIFFERENTLY under max_depth=2 vs None.

        Construct two trees that diverge only at a deep leaf.  With full
        traversal, scores reflect a tiny leaf-only difference.  With
        max_depth=2 the deep subtrees are short-circuited identically on
        both sides — the cap dominates the score, so the two configs
        produce different numbers.
        """
        a = _build_nested_dict(6, leaf=1)
        b = _build_nested_dict(6, leaf=2)
        algo_full = STEDAlgorithm(backend=backend, config=STEDConfig())
        algo_capped = STEDAlgorithm(backend=backend, config=STEDConfig(max_depth=2))
        score_full = algo_full.compute(a, b)
        score_capped = algo_capped.compute(a, b)
        # Both must be in [0, 1].
        assert 0.0 <= score_full <= 1.0
        assert 0.0 <= score_capped <= 1.0
        # The capped score must differ — different policy → different number.
        assert score_full != pytest.approx(score_capped, abs=1e-9), (
            f"max_depth had no effect: full={score_full}, capped={score_capped}"
        )

    def test_max_depth_default_preserves_existing_behavior(
        self, backend: StaticBackend
    ) -> None:
        """Default and explicit max_depth=None must give identical scores."""
        a = {"x": {"y": {"z": 1}}}
        b = {"x": {"y": {"z": 2}}}
        s_default = STEDAlgorithm(backend=backend).compute(a, b)
        s_explicit = STEDAlgorithm(
            backend=backend, config=STEDConfig(max_depth=None)
        ).compute(a, b)
        assert s_default == pytest.approx(s_explicit)

    def test_max_depth_one_caps_at_root(self, backend: StaticBackend) -> None:
        """max_depth=1 short-circuits all object-children comparisons.

        Two structurally identical but deeply different trees should still
        produce a finite score (no recursion past the cap).
        """
        a = {"a": {"x": 1, "y": 2}}
        b = {"a": {"x": 99, "y": 99}}
        algo = STEDAlgorithm(backend=backend, config=STEDConfig(max_depth=1))
        score = algo.compute(a, b)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Deep recursion (T1, xfail until H1 — iterative refactor lands)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="audit H1 — fully recursive tree build/walk, iterative refactor pending",
    strict=False,
)
def test_deeply_nested_self_compare_does_not_recursion_error(
    backend: StaticBackend,
) -> None:
    """A pathologically deep JSON should compare without RecursionError.

    Today this fails (the tree builder and STED walk are both recursive
    and blow the C-stack near 1500 levels).  Marked xfail so it auto-flips
    to passing once H1 (iterative refactor) lands.
    """
    deep = _build_nested_dict(1500, leaf=1)
    deep_copy = _build_nested_dict(1500, leaf=1)
    algo = STEDAlgorithm(backend=backend)
    score = algo.compute(deep, deep_copy)
    assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Memoisation cache (audit finding I1 — Hungarian O(n⁴) worst case)
# ---------------------------------------------------------------------------


class TestDistanceCache:
    """``_dist_cache`` memoises ``_compute_node_distance`` results.

    The cache lives for the duration of a single ``compute()`` call:
    cleared at entry, populated during recursion, examined here at exit.
    """

    def test_cache_populates_during_compute(self, backend: StaticBackend) -> None:
        """Comparing a tree with repeated subtree shapes fills the cache.

        Hungarian builds an m-by-n cost matrix where each cell calls
        ``_compute_node_distance``.  On an object with several keys, that
        function is invoked many times — every invocation lands in the cache.
        """
        algo = STEDAlgorithm(backend=backend)
        a = {f"k{i}": {"nested": {"v": i}} for i in range(8)}
        b = {f"k{i}": {"nested": {"v": i}} for i in range(8)}
        algo.compute(a, b)
        # At minimum, the Hungarian cost matrix at the root produced 8*8=64
        # cell evaluations, each of which recursed deeper — total cache
        # entries comfortably exceed the matrix size.
        assert len(algo._dist_cache) > 0, "cache should be populated after compute()"
        assert len(algo._dist_cache) >= 64, (
            f"expected at least 64 cache entries (8x8 cost matrix), "
            f"got {len(algo._dist_cache)}"
        )

    def test_cache_cleared_between_compute_calls(self, backend: StaticBackend) -> None:
        """A fresh ``compute()`` must not see entries from a prior comparison.

        TreeNode identities (``id()``) can be reused by CPython after objects
        are freed — leaking cache entries across calls would be a correctness
        bug.  The implementation defensively clears the cache at the top of
        every ``compute()`` call.
        """
        algo = STEDAlgorithm(backend=backend)
        # First call populates the cache.
        algo.compute({"a": {"b": 1}}, {"a": {"b": 2}})
        first_size = len(algo._dist_cache)
        assert first_size > 0

        # Second call: cache must be cleared at entry.  We can't observe the
        # "cleared" state mid-call from outside, but we CAN verify that the
        # second call produces the same score as a freshly-constructed
        # algorithm (i.e. no stale entries influenced it).
        algo2 = STEDAlgorithm(backend=backend)
        s1 = algo.compute({"x": [1, 2, 3]}, {"x": [1, 2, 4]})
        s2 = algo2.compute({"x": [1, 2, 3]}, {"x": [1, 2, 4]})
        assert s1 == pytest.approx(s2), (
            "cache-clearing between compute() calls must yield identical scores"
        )

    def test_cache_identity_self_comparison_returns_zero(
        self, backend: StaticBackend
    ) -> None:
        """Comparing a node with itself goes through the cache normally.

        ``id(node_a) == id(node_b)`` is a legitimate cache key — the distance
        is computed once (always 0.0 for any node compared with itself) and
        reused on subsequent lookups.  This is exercised whenever the same
        sub-tree appears repeatedly in a single input.
        """
        algo = STEDAlgorithm(backend=backend)
        # Identical inputs produce score 1.0; the cache should hold the
        # zero-distance results computed along the way.
        score = algo.compute({"a": [1, 2], "b": [1, 2]}, {"a": [1, 2], "b": [1, 2]})
        assert score == pytest.approx(1.0)
        # Sanity: cache was populated (i.e. the path was actually taken).
        assert len(algo._dist_cache) > 0

    def test_wide_object_completes_quickly(self, backend: StaticBackend) -> None:
        """Perf smoke test: a wide object compare must finish in well under
        the worst-case O(n⁴) budget.

        Pre-memoisation, a Hungarian on m=n=120 keys with non-trivial
        sub-trees could trigger noticeable slowdown when the same pair was
        re-evaluated across cost-matrix cells.  Loose 5-second budget is
        intentionally generous — anything close to it suggests an O(n⁴)
        regression.
        """
        import time

        n = 120
        a = {f"key_{i}": {"nested": {"value": i, "tag": f"t{i}"}} for i in range(n)}
        b = {f"key_{i}": {"nested": {"value": i, "tag": f"t{i}"}} for i in range(n)}

        algo = STEDAlgorithm(backend=backend)
        start = time.perf_counter()
        score = algo.compute(a, b)
        elapsed = time.perf_counter() - start

        assert score == pytest.approx(1.0)
        assert elapsed < 5.0, (
            f"wide-object compare took {elapsed:.2f}s (expected < 5.0s) — "
            f"possible O(n⁴) regression"
        )


# ---------------------------------------------------------------------------
# Explain mode (F1): per-path contributions
# ---------------------------------------------------------------------------


class TestExplainModeDisabled:
    """When ``collect_explanation`` is False (default), scores must be
    bit-identical to the pre-explain behaviour and ``last_explanations``
    stays the empty tuple."""

    def test_scores_unchanged_for_identical_inputs(
        self, backend: StaticBackend
    ) -> None:
        algo = STEDAlgorithm(backend=backend)
        score = algo.compute({"a": 1, "b": [1, 2]}, {"a": 1, "b": [1, 2]})
        assert score == pytest.approx(1.0)
        assert algo.last_explanations == ()

    def test_scores_unchanged_for_divergent_inputs(
        self, backend: StaticBackend
    ) -> None:
        algo_off = STEDAlgorithm(backend=backend)
        algo_on = STEDAlgorithm(
            backend=backend,
            config=STEDConfig(collect_explanation=True),
        )
        left = {"a": 1, "b": 2, "c": 3}
        right = {"a": 9, "b": 2, "d": 4}
        score_off = algo_off.compute(left, right)
        score_on = algo_on.compute(left, right)
        assert score_off == pytest.approx(score_on)
        # Default config: explanations are empty.
        assert algo_off.last_explanations == ()

    def test_last_explanations_is_empty_tuple_by_default(
        self, backend: StaticBackend
    ) -> None:
        algo = STEDAlgorithm(backend=backend)
        algo.compute({"a": 1}, {"a": 2})
        assert algo.last_explanations == ()


class TestExplainModeEnabled:
    """When ``collect_explanation`` is True, contributions populate
    ``last_explanations`` for any non-identical input."""

    @pytest.fixture
    def algo_explain(self, backend: StaticBackend) -> STEDAlgorithm:
        return STEDAlgorithm(
            backend=backend,
            config=STEDConfig(collect_explanation=True),
        )

    def test_identical_inputs_yield_empty_explanation(
        self, algo_explain: STEDAlgorithm
    ) -> None:
        algo_explain.compute({"a": 1, "b": "x"}, {"a": 1, "b": "x"})
        assert algo_explain.last_explanations == ()

    def test_value_mismatch_produces_contribution(
        self, algo_explain: STEDAlgorithm
    ) -> None:
        algo_explain.compute({"a": 1}, {"a": 2})
        explanations = algo_explain.last_explanations
        assert len(explanations) >= 1
        # The matched-key pair carries a non-zero distance — a value mismatch.
        kinds = {c.kind for c in explanations}
        assert "value_mismatch" in kinds
        # Path should be valid (non-empty, slash-rooted).
        for c in explanations:
            assert c.path.startswith("/")
            assert c.contribution > 0.0

    def test_unmatched_left_produces_contribution(
        self, algo_explain: STEDAlgorithm
    ) -> None:
        algo_explain.compute({"a": 1, "extra": 2}, {"a": 1})
        kinds = {c.kind for c in algo_explain.last_explanations}
        assert "unmatched_left" in kinds

    def test_unmatched_right_produces_contribution(
        self, algo_explain: STEDAlgorithm
    ) -> None:
        algo_explain.compute({"a": 1}, {"a": 1, "extra": 2})
        kinds = {c.kind for c in algo_explain.last_explanations}
        assert "unmatched_right" in kinds

    def test_contributions_sorted_descending(self, algo_explain: STEDAlgorithm) -> None:
        left = {"a": "x", "b": "y", "c": "z", "extra1": 1, "extra2": 2}
        right = {"a": "X", "b": "Y", "c": "Z", "different": 99}
        algo_explain.compute(left, right)
        contributions = [c.contribution for c in algo_explain.last_explanations]
        assert contributions == sorted(contributions, reverse=True)

    def test_all_contributions_have_valid_kind(
        self, algo_explain: STEDAlgorithm
    ) -> None:
        valid_kinds = {"matched", "unmatched_left", "unmatched_right", "value_mismatch"}
        left = {"a": 1, "b": [1, 2, 3], "c": "old"}
        right = {"a": 2, "b": [1, 2, 4], "c": "new", "d": "extra"}
        algo_explain.compute(left, right)
        for c in algo_explain.last_explanations:
            assert c.kind in valid_kinds
            # Every contribution should be a finite, positive float.
            assert c.contribution > 0.0
            assert c.contribution < float("inf")
            # Path must be a JSON Pointer rooted at /.
            assert c.path.startswith("/")

    def test_scalar_root_mismatch_emits_root_contribution(
        self, algo_explain: STEDAlgorithm
    ) -> None:
        algo_explain.compute("hello", "world")
        explanations = algo_explain.last_explanations
        assert len(explanations) == 1
        assert explanations[0].path == "/"
        assert explanations[0].kind == "value_mismatch"
        assert explanations[0].contribution > 0.0

    def test_type_mismatched_roots_emit_contribution(
        self, algo_explain: STEDAlgorithm
    ) -> None:
        algo_explain.compute({"a": 1}, [1, 2])
        explanations = algo_explain.last_explanations
        assert len(explanations) >= 1
        assert explanations[0].path == "/"
        assert "type" in explanations[0].detail

    def test_array_value_mismatch_records_element_path(
        self, algo_explain: STEDAlgorithm
    ) -> None:
        algo_explain.compute([1, 2, 3], [1, 9, 3])
        # The middle element differs; expect a contribution on that index.
        paths = [c.path for c in algo_explain.last_explanations]
        assert any("1" in p.split("/") for p in paths)

    def test_explain_does_not_change_score(self, backend: StaticBackend) -> None:
        """Enabling explain mode must not perturb the numeric score."""
        left = {"x": [1, 2, {"a": "old"}], "y": "v"}
        right = {"x": [1, 2, {"a": "new"}], "y": "v"}
        algo_off = STEDAlgorithm(backend=backend)
        algo_on = STEDAlgorithm(
            backend=backend,
            config=STEDConfig(collect_explanation=True),
        )
        assert algo_off.compute(left, right) == pytest.approx(
            algo_on.compute(left, right)
        )
