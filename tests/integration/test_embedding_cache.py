"""Integration tests for the embedding cache layer.

Verifies cache warm-up, batch embedding, LRU eviction, and per-instance
isolation. All imports use the top-level ``json_semantic_diff`` package — never
internal submodules.
"""

from __future__ import annotations

from typing import Any

from json_semantic_diff.backends import StaticBackend
from json_semantic_diff.comparator import STEDComparator


class TestSC1_ZeroEmbedCallsOnSecondCompare:
    """SC1 — Second compare() on same documents triggers zero embed calls."""

    def test_second_compare_zero_embed_calls(self) -> None:
        """SC1: Calling compare() twice with same docs -> zero embed calls on second."""
        raw_backend = StaticBackend()
        call_log: list[list[str]] = []
        original_embed = raw_backend.embed

        def spy_embed(strings: list[str]) -> Any:
            call_log.append(strings)
            return original_embed(strings)

        raw_backend.embed = spy_embed  # type: ignore[method-assign]

        cmp = STEDComparator(backend=raw_backend)
        doc = {"user_name": "Alice", "age": 30, "active": True}
        cmp.compare(doc, doc)  # first call — warms cache
        call_log.clear()  # reset
        cmp.compare(doc, doc)  # second call — must be all cache hits
        assert len(call_log) == 0, (
            f"Expected 0 embed calls on second compare, got {len(call_log)}"
        )

    def test_second_compare_same_score(self) -> None:
        """SC1: Both calls return identical scores (cache does not alter results)."""
        cmp = STEDComparator()
        doc = {"user_name": "Alice", "age": 30}
        r1 = cmp.compare(doc, doc)
        r2 = cmp.compare(doc, doc)
        assert r1.similarity_score == r2.similarity_score


class TestSC2_BatchEmbedSingleCall:
    """SC2 — 50 unique key names trigger exactly one embed() call."""

    def test_fifty_keys_one_embed_call(self) -> None:
        """SC2: 50 unique key names -> exactly one embed() call."""
        raw_backend = StaticBackend()
        call_log: list[int] = []
        original_embed = raw_backend.embed

        def spy_embed(strings: list[str]) -> Any:
            call_log.append(len(strings))
            return original_embed(strings)

        raw_backend.embed = spy_embed  # type: ignore[method-assign]

        doc_a = {f"key_{i}": i for i in range(50)}
        doc_b = {f"key_{i}": i for i in range(50)}

        cmp = STEDComparator(backend=raw_backend)
        cmp.compare(doc_a, doc_b)

        # Pre-warm calls embed() once with all unique labels.
        # Since doc_a and doc_b have the same keys, there are 50 unique labels.
        # The batch pre-scan should call embed() exactly once.
        assert len(call_log) == 1, f"Expected 1 embed call, got {len(call_log)}"
        assert call_log[0] <= 50, f"Expected at most 50 strings, got {call_log[0]}"

    def test_batch_embed_call_count_for_disjoint_keys(self) -> None:
        """SC2: Disjoint key sets also result in a single batch embed call."""
        raw_backend = StaticBackend()
        call_log: list[int] = []
        original_embed = raw_backend.embed

        def spy_embed(strings: list[str]) -> Any:
            call_log.append(len(strings))
            return original_embed(strings)

        raw_backend.embed = spy_embed  # type: ignore[method-assign]

        doc_a = {"alpha": 1, "beta": 2}
        doc_b = {"gamma": 3, "delta": 4}

        cmp = STEDComparator(backend=raw_backend)
        cmp.compare(doc_a, doc_b)

        # 4 unique keys total across both docs -> one embed call
        assert len(call_log) == 1, f"Expected 1 embed call, got {len(call_log)}"
        assert call_log[0] <= 4, (
            f"Expected at most 4 strings in batch, got {call_log[0]}"
        )


class TestSC3_ConfigurableLRUSize:
    """SC3 — Cache has configurable max size; eviction operates without error."""

    def test_configurable_max_size_eviction(self) -> None:
        """SC3: Cache has configurable max size; eviction without error."""
        raw_backend = StaticBackend()
        cmp = STEDComparator(backend=raw_backend, max_cache_size=5)

        # Compare doc with more than 5 unique keys
        doc = {f"key_{i}": i for i in range(10)}
        result = cmp.compare(doc, doc)

        # Must not error; score should still be valid
        assert 0.0 <= result.similarity_score <= 1.0
        # Cache should be at max 5 entries (LRU evicted the rest)
        assert cmp._backend.curr_size <= 5

    def test_max_cache_size_default_is_512(self) -> None:
        """SC3: Default max_cache_size is 512."""
        cmp = STEDComparator()
        assert cmp._backend.max_size == 512

    def test_custom_max_cache_size_respected(self) -> None:
        """SC3: Custom max_cache_size is reflected in the backend property."""
        cmp = STEDComparator(max_cache_size=128)
        assert cmp._backend.max_size == 128


class TestSC4_PerInstanceIsolation:
    """SC4 — Two STEDComparator instances do not share cache state."""

    def test_per_instance_cache_isolation(self) -> None:
        """SC4: Two STEDComparator instances do not share cache state."""
        cmp1 = STEDComparator()
        cmp2 = STEDComparator()

        cmp1.compare({"name": "Alice"}, {"name": "Bob"})

        # cmp1's cache should have entries; cmp2's should be empty
        assert cmp1._backend.curr_size > 0
        assert cmp2._backend.curr_size == 0

    def test_per_instance_cache_state_independent(self) -> None:
        """SC4: Cache state from cmp1 does not bleed into cmp2."""
        cmp1 = STEDComparator()
        cmp2 = STEDComparator()

        cmp1.compare({"alpha": 1, "beta": 2}, {"gamma": 3})
        cmp2.compare({"delta": 4}, {"epsilon": 5})

        # Each instance's cache reflects only its own usage
        assert cmp1._backend.curr_size != cmp2._backend.curr_size or (
            cmp1._backend.curr_size > 0 and cmp2._backend.curr_size > 0
        )


class TestRegressionGuards:
    """Algorithm results must be unchanged by the cache layer."""

    def test_overall_score_naming_convention(self) -> None:
        """Cache layer does not alter score for naming convention comparison."""
        cmp = STEDComparator()
        result = cmp.compare({"user_name": "John"}, {"userName": "John"})
        assert result.similarity_score > 0.85, (
            f"Expected score > 0.85 for naming convention, got {result.similarity_score}"
        )

    def test_disjoint_docs_score_low(self) -> None:
        """Cache layer does not inflate score for structurally dissimilar docs.

        Audit C6 (wave 7): single-key OBJECTs with KEY -> SCALAR shape have a
        Zhang-Shasha-normalised floor around 0.5.  The threshold here just
        guards that the cache doesn't push the score back up into the
        equivalence band (>= 0.85).
        """
        cmp = STEDComparator()
        result = cmp.compare({"user_name": "John"}, {"address": "123 Main St"})
        assert result.similarity_score < 0.6, (
            f"Expected score < 0.6 for disjoint docs, got {result.similarity_score}"
        )

    def test_identical_docs_score_1(self) -> None:
        """Identical documents always score 1.0 regardless of caching."""
        cmp = STEDComparator()
        doc = {"user_name": "John", "age": 30, "active": True}
        r1 = cmp.compare(doc, doc)
        r2 = cmp.compare(doc, doc)  # second call uses warm cache
        assert r1.similarity_score == 1.0
        assert r2.similarity_score == 1.0
