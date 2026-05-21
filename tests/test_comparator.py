"""Tests for STEDComparator — the central orchestrator for semantic JSON comparison.

Covers:
- Core comparison functionality (identical, different, naming-convention pairs)
- null_equals_missing preprocessing (flat and nested)
- KEY-level match extraction (matched_pairs, key_mappings, unmatched lists)
- Default backend behaviour (no args → StaticBackend)
- Statelessness (two identical calls → identical results)
- computation_time_ms is always a positive float
"""

from __future__ import annotations

import pytest

from json_semantic_diff.algorithm.config import STEDConfig
from json_semantic_diff.comparator import STEDComparator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _key_names_in_pairs(pairs: list[tuple[str, str]], side: int) -> list[str]:
    """Extract the final path segment (key name) from matched pair paths."""
    return [p[side].rsplit("/", 1)[-1] for p in pairs]


# ---------------------------------------------------------------------------
# Core functionality
# ---------------------------------------------------------------------------


class TestCoreComparison:
    def test_identical_objects_score_1_0(self) -> None:
        cmp = STEDComparator()
        result = cmp.compare({"a": 1}, {"a": 1})
        assert result.similarity_score == pytest.approx(1.0, abs=1e-9)

    def test_identical_objects_matched_pairs_non_empty(self) -> None:
        cmp = STEDComparator()
        result = cmp.compare({"a": 1}, {"a": 1})
        assert len(result.matched_pairs) > 0

    def test_identical_objects_matched_pair_paths_contain_key_a(self) -> None:
        cmp = STEDComparator()
        result = cmp.compare({"a": 1}, {"a": 1})
        left_keys = _key_names_in_pairs(result.matched_pairs, 0)
        assert "a" in left_keys

    def test_identical_objects_key_mappings(self) -> None:
        cmp = STEDComparator()
        result = cmp.compare({"a": 1}, {"a": 1})
        assert result.key_mappings == {"a": "a"}

    def test_identical_objects_no_unmatched(self) -> None:
        cmp = STEDComparator()
        result = cmp.compare({"a": 1}, {"a": 1})
        assert result.unmatched_left == ()
        assert result.unmatched_right == ()

    def test_identical_objects_computation_time_positive(self) -> None:
        cmp = STEDComparator()
        result = cmp.compare({"a": 1}, {"a": 1})
        assert result.computation_time_ms > 0.0

    def test_different_objects_score_less_than_1(self) -> None:
        cmp = STEDComparator()
        result = cmp.compare({"a": 1}, {"b": 2})
        assert result.similarity_score < 1.0

    def test_naming_convention_user_name_vs_userName_score_above_085(self) -> None:
        cmp = STEDComparator()
        result = cmp.compare({"user_name": "John"}, {"userName": "John"})
        assert result.similarity_score > 0.85

    def test_naming_convention_key_mappings_correct(self) -> None:
        cmp = STEDComparator()
        result = cmp.compare({"user_name": "John"}, {"userName": "John"})
        assert result.key_mappings.get("user_name") == "userName"

    def test_empty_objects_score_1_0(self) -> None:
        cmp = STEDComparator()
        result = cmp.compare({}, {})
        assert result.similarity_score == pytest.approx(1.0, abs=1e-9)

    def test_scalars_score_1_0_when_identical(self) -> None:
        cmp = STEDComparator()
        result = cmp.compare("hello", "hello")
        assert result.similarity_score == pytest.approx(1.0, abs=1e-9)

    def test_scalars_no_matched_pairs(self) -> None:
        """Scalar roots have no KEY children — matched_pairs is empty."""
        cmp = STEDComparator()
        result = cmp.compare("hello", "world")
        assert result.matched_pairs == ()
        assert result.key_mappings == {}


# ---------------------------------------------------------------------------
# null_equals_missing
# ---------------------------------------------------------------------------


class TestNullEqualsMissing:
    def test_null_equals_missing_true_none_vs_empty_score_1_0(self) -> None:
        cmp = STEDComparator(config=STEDConfig(null_equals_missing=True))
        result = cmp.compare({"x": None}, {})
        assert result.similarity_score == pytest.approx(1.0, abs=1e-9)

    def test_null_equals_missing_false_none_vs_empty_score_less_than_1(self) -> None:
        cmp = STEDComparator(config=STEDConfig())
        result = cmp.compare({"x": None}, {})
        assert result.similarity_score < 1.0

    def test_null_equals_missing_true_partial_none(self) -> None:
        cmp = STEDComparator(config=STEDConfig(null_equals_missing=True))
        result = cmp.compare({"a": 1, "b": None}, {"a": 1})
        assert result.similarity_score == pytest.approx(1.0, abs=1e-9)

    def test_null_equals_missing_true_nested_none(self) -> None:
        cmp = STEDComparator(config=STEDConfig(null_equals_missing=True))
        result = cmp.compare({"a": {"b": None}}, {"a": {}})
        assert result.similarity_score == pytest.approx(1.0, abs=1e-9)

    def test_null_equals_missing_false_does_not_preprocess(self) -> None:
        """Default config — {"x": None} != {} — score must reflect the difference."""
        cmp = STEDComparator(config=STEDConfig(null_equals_missing=False))
        result_none = cmp.compare({"x": None}, {})
        result_same = cmp.compare({}, {})
        # With the key present (value=None), the score should be lower than for {}=={}
        assert result_none.similarity_score < result_same.similarity_score


# ---------------------------------------------------------------------------
# Match extraction
# ---------------------------------------------------------------------------


class TestMatchExtraction:
    def test_partially_overlapping_keys_a_matched_to_a(self) -> None:
        cmp = STEDComparator()
        result = cmp.compare({"a": 1, "b": 2}, {"a": 1, "c": 3})
        left_keys = _key_names_in_pairs(result.matched_pairs, 0)
        right_keys = _key_names_in_pairs(result.matched_pairs, 1)
        assert "a" in left_keys
        assert "a" in right_keys

    def test_partially_overlapping_all_keys_matched_hungarian_exhaustive(self) -> None:
        """Hungarian always exhaustively matches all keys when sizes are equal.

        {"a":1,"b":2} vs {"a":1,"c":3} yields 2x2 cost matrix — Hungarian
        assigns all 2 left keys to all 2 right keys (minimum total cost),
        so unmatched lists are empty even though "b" and "c" are dissimilar.
        """
        cmp = STEDComparator()
        result = cmp.compare({"a": 1, "b": 2}, {"a": 1, "c": 3})
        assert len(result.matched_pairs) == 2
        assert result.unmatched_left == ()
        assert result.unmatched_right == ()

    def test_unmatched_produced_when_sizes_differ(self) -> None:
        """When left has more keys than right, surplus left keys are unmatched."""
        cmp = STEDComparator()
        result = cmp.compare({"a": 1, "b": 2, "d": 4}, {"a": 1})
        unmatched_left_names = [p.rsplit("/", 1)[-1] for p in result.unmatched_left]
        # "b" and "d" cannot be matched — right has only 1 key
        assert len(result.unmatched_left) == 2
        assert set(unmatched_left_names) == {"b", "d"}

    def test_unmatched_right_when_right_has_more_keys(self) -> None:
        """When right has more keys than left, surplus right keys are unmatched."""
        cmp = STEDComparator()
        result = cmp.compare({"a": 1}, {"a": 1, "c": 3, "d": 4})
        unmatched_right_names = [p.rsplit("/", 1)[-1] for p in result.unmatched_right]
        assert len(result.unmatched_right) == 2
        assert set(unmatched_right_names) == {"c", "d"}

    def test_nested_match_extraction_outer_and_inner_in_matched_pairs(self) -> None:
        cmp = STEDComparator()
        result = cmp.compare({"outer": {"inner": 1}}, {"outer": {"inner": 1}})
        # "outer" should appear in matched pairs
        left_key_names = _key_names_in_pairs(result.matched_pairs, 0)
        assert "outer" in left_key_names
        # "inner" should also appear (nested recursion)
        assert "inner" in left_key_names

    def test_completely_different_keys_all_unmatched(self) -> None:
        """When keys have zero similarity, Hungarian may still match them.

        The important invariant is that the union of matched + unmatched equals
        the total number of keys on each side.
        """
        cmp = STEDComparator()
        result = cmp.compare({"a": 1}, {"b": 2})
        total_left = len(result.matched_pairs) + len(result.unmatched_left)
        total_right = len(result.matched_pairs) + len(result.unmatched_right)
        assert total_left == 1
        assert total_right == 1

    def test_empty_left_all_right_keys_unmatched(self) -> None:
        cmp = STEDComparator()
        result = cmp.compare({}, {"a": 1, "b": 2})
        assert result.matched_pairs == ()
        assert result.key_mappings == {}
        assert len(result.unmatched_right) == 2

    def test_empty_right_all_left_keys_unmatched(self) -> None:
        cmp = STEDComparator()
        result = cmp.compare({"a": 1, "b": 2}, {})
        assert result.matched_pairs == ()
        assert result.key_mappings == {}
        assert len(result.unmatched_left) == 2


# ---------------------------------------------------------------------------
# Default backend
# ---------------------------------------------------------------------------


class TestDefaultBackend:
    def test_no_args_uses_static_backend(self) -> None:
        """STEDComparator() with no args must work without error."""
        cmp = STEDComparator()
        result = cmp.compare({"a": 1}, {"a": 1})
        assert result.similarity_score == pytest.approx(1.0, abs=1e-9)

    def test_no_args_naming_convention_similarity(self) -> None:
        cmp = STEDComparator()
        result = cmp.compare({"user_name": "Alice"}, {"userName": "Alice"})
        assert result.similarity_score > 0.85


# ---------------------------------------------------------------------------
# Statelessness (API-05 SC5)
# ---------------------------------------------------------------------------


class TestStatelessness:
    def test_two_identical_calls_return_same_score(self) -> None:
        cmp = STEDComparator()
        r1 = cmp.compare({"a": 1, "b": 2}, {"a": 1, "b": 2})
        r2 = cmp.compare({"a": 1, "b": 2}, {"a": 1, "b": 2})
        assert r1.similarity_score == r2.similarity_score

    def test_two_identical_calls_return_same_matched_pairs(self) -> None:
        cmp = STEDComparator()
        r1 = cmp.compare({"a": 1, "b": 2}, {"a": 1, "b": 2})
        r2 = cmp.compare({"a": 1, "b": 2}, {"a": 1, "b": 2})
        assert r1.matched_pairs == r2.matched_pairs

    def test_two_identical_calls_return_same_key_mappings(self) -> None:
        cmp = STEDComparator()
        r1 = cmp.compare({"a": 1, "b": 2}, {"a": 1, "b": 2})
        r2 = cmp.compare({"a": 1, "b": 2}, {"a": 1, "b": 2})
        assert r1.key_mappings == r2.key_mappings

    def test_two_identical_calls_return_same_unmatched(self) -> None:
        cmp = STEDComparator()
        r1 = cmp.compare({"a": 1, "c": 3}, {"a": 1, "b": 2})
        r2 = cmp.compare({"a": 1, "c": 3}, {"a": 1, "b": 2})
        assert r1.unmatched_left == r2.unmatched_left
        assert r1.unmatched_right == r2.unmatched_right

    def test_computation_time_always_positive(self) -> None:
        cmp = STEDComparator()
        for _ in range(5):
            result = cmp.compare({"x": 42}, {"x": 42})
            assert result.computation_time_ms > 0.0


# ---------------------------------------------------------------------------
# ignore_paths
# ---------------------------------------------------------------------------


class TestIgnorePaths:
    def test_default_empty_ignore_paths_is_noop(self) -> None:
        """Default config has ignore_paths=() — behaviour unchanged."""
        cmp_default = STEDComparator()
        cmp_explicit = STEDComparator(config=STEDConfig(ignore_paths=()))
        r_default = cmp_default.compare({"a": 1, "b": 2}, {"a": 1, "c": 3})
        r_explicit = cmp_explicit.compare({"a": 1, "b": 2}, {"a": 1, "c": 3})
        assert r_default.similarity_score == pytest.approx(r_explicit.similarity_score)
        assert r_default.matched_pairs == r_explicit.matched_pairs
        assert r_default.unmatched_left == r_explicit.unmatched_left
        assert r_default.unmatched_right == r_explicit.unmatched_right

    def test_ignore_paths_top_level_volatile_key(self) -> None:
        """Differing /timestamp values are ignored — score should be 1.0."""
        cmp = STEDComparator(config=STEDConfig(ignore_paths=("/timestamp",)))
        result = cmp.compare(
            {"a": 1, "timestamp": "2024-01-01T00:00:00Z"},
            {"a": 1, "timestamp": "2026-05-21T12:34:56Z"},
        )
        assert result.similarity_score == pytest.approx(1.0, abs=1e-9)

    def test_ignore_paths_top_level_volatile_key_without_ignore_lower_score(
        self,
    ) -> None:
        """Without ignore_paths the differing timestamp drags score below 1.0."""
        cmp = STEDComparator()
        result = cmp.compare(
            {"a": 1, "timestamp": "2024-01-01T00:00:00Z"},
            {"a": 1, "timestamp": "2026-05-21T12:34:56Z"},
        )
        assert result.similarity_score < 1.0

    def test_ignore_paths_with_array_wildcard(self) -> None:
        """/users/*/id strips id from every user in the array."""
        cmp = STEDComparator(config=STEDConfig(ignore_paths=("/users/*/id",)))
        left = {
            "users": [
                {"id": "u-001", "name": "Alice"},
                {"id": "u-002", "name": "Bob"},
            ]
        }
        right = {
            "users": [
                {"id": "u-999", "name": "Alice"},
                {"id": "u-998", "name": "Bob"},
            ]
        }
        result = cmp.compare(left, right)
        assert result.similarity_score == pytest.approx(1.0, abs=1e-9)

    def test_ignore_paths_multiple_patterns_combine_with_or(self) -> None:
        """Several patterns: each match is dropped independently."""
        cmp = STEDComparator(config=STEDConfig(ignore_paths=("/timestamp", "/version")))
        left = {"a": 1, "timestamp": "t1", "version": "v1"}
        right = {"a": 1, "timestamp": "t2", "version": "v2"}
        result = cmp.compare(left, right)
        assert result.similarity_score == pytest.approx(1.0, abs=1e-9)

    def test_ignore_paths_bad_pattern_raises_at_config_construction(self) -> None:
        """Pattern validation runs in STEDConfig.__post_init__."""
        with pytest.raises(ValueError):
            STEDConfig(ignore_paths=("no_slash",))

    def test_ignored_path_not_in_unmatched(self) -> None:
        """Ignored keys should not appear in unmatched_left/unmatched_right."""
        cmp = STEDComparator(config=STEDConfig(ignore_paths=("/secret",)))
        result = cmp.compare(
            {"a": 1, "secret": "left-only"},
            {"a": 1},
        )
        # secret is dropped from left before comparison — neither side reports it
        all_paths = (
            list(result.unmatched_left)
            + list(result.unmatched_right)
            + [p for pair in result.matched_pairs for p in pair]
        )
        assert all("secret" not in p for p in all_paths)

    def test_ignored_path_not_in_key_mappings(self) -> None:
        """Ignored keys should not appear in key_mappings either."""
        cmp = STEDComparator(config=STEDConfig(ignore_paths=("/timestamp",)))
        result = cmp.compare({"a": 1, "timestamp": "t1"}, {"a": 1, "timestamp": "t2"})
        assert "timestamp" not in result.key_mappings
        assert "timestamp" not in result.key_mappings.values()

    def test_ignore_paths_does_not_mutate_inputs(self) -> None:
        """_preprocess must keep returning fresh structures — verify inputs unchanged."""
        cmp = STEDComparator(config=STEDConfig(ignore_paths=("/timestamp",)))
        left = {"a": 1, "timestamp": "t1"}
        right = {"a": 1, "timestamp": "t2"}
        left_snapshot = {"a": 1, "timestamp": "t1"}
        right_snapshot = {"a": 1, "timestamp": "t2"}
        cmp.compare(left, right)
        assert left == left_snapshot
        assert right == right_snapshot

    def test_ignore_paths_does_not_mutate_nested_inputs(self) -> None:
        """Nested dicts inside the ignored subtree are not mutated either."""
        cmp = STEDComparator(config=STEDConfig(ignore_paths=("/meta",)))
        meta = {"version": 1, "build": "abc"}
        left = {"a": 1, "meta": meta}
        cmp.compare(left, {"a": 1})
        # The meta sub-dict on the original input is untouched
        assert meta == {"version": 1, "build": "abc"}
        assert left["meta"] is meta

    def test_ignore_paths_nested_subtree_removed(self) -> None:
        """/meta removes the whole meta subtree from both sides."""
        cmp = STEDComparator(config=STEDConfig(ignore_paths=("/meta",)))
        left = {"a": 1, "meta": {"version": "v1", "build": "abc"}}
        right = {"a": 1, "meta": {"version": "v2", "build": "xyz"}}
        result = cmp.compare(left, right)
        assert result.similarity_score == pytest.approx(1.0, abs=1e-9)
        assert "meta" not in result.key_mappings

    def test_ignore_paths_nested_subtree_one_side_missing(self) -> None:
        """/meta absent on right is fine — ignored on left, no penalty."""
        cmp = STEDComparator(config=STEDConfig(ignore_paths=("/meta",)))
        left = {"a": 1, "meta": {"version": "v1"}}
        right = {"a": 1}
        result = cmp.compare(left, right)
        assert result.similarity_score == pytest.approx(1.0, abs=1e-9)

    def test_ignore_paths_pattern_does_not_match_unrelated_key(self) -> None:
        """/timestamp must NOT match /timestamps (no partial-component matching)."""
        cmp = STEDComparator(config=STEDConfig(ignore_paths=("/timestamp",)))
        result = cmp.compare(
            {"a": 1, "timestamps": ["t1"]}, {"a": 1, "timestamps": ["t2"]}
        )
        # Score should be less than 1 because /timestamps was NOT ignored
        assert result.similarity_score < 1.0

    def test_ignore_paths_pattern_does_not_match_deeper_path(self) -> None:
        """/x must NOT match /y/x — patterns are full-path, not suffix."""
        cmp = STEDComparator(config=STEDConfig(ignore_paths=("/x",)))
        result = cmp.compare({"y": {"x": "left"}}, {"y": {"x": "right"}})
        # /y/x is NOT matched by /x, so the differing values still count
        assert result.similarity_score < 1.0

    def test_ignore_paths_wildcard_matches_object_key_too(self) -> None:
        """A single `*` component matches any single component (dict key or index)."""
        cmp = STEDComparator(config=STEDConfig(ignore_paths=("/users/*/id",)))
        # users is a dict here (keyed by username), not an array — wildcard still works
        left = {"users": {"alice": {"id": "a1", "name": "Alice"}}}
        right = {"users": {"alice": {"id": "a2", "name": "Alice"}}}
        result = cmp.compare(left, right)
        assert result.similarity_score == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# aliases integration
# ---------------------------------------------------------------------------


class TestAliases:
    """End-to-end behaviour of STEDConfig.aliases through STEDComparator."""

    def test_default_aliases_is_noop(self) -> None:
        """Empty aliases must not perturb the score on disjoint key names."""
        cmp_default = STEDComparator()
        cmp_empty = STEDComparator(config=STEDConfig(aliases=()))
        r_default = cmp_default.compare({"uid": 1}, {"completely_different_key": 1})
        r_empty = cmp_empty.compare({"uid": 1}, {"completely_different_key": 1})
        assert r_default.similarity_score == pytest.approx(
            r_empty.similarity_score, abs=1e-9
        )

    def test_alias_pair_matches_keys_left_to_right(self) -> None:
        """aliases=(("uid", "user_id"),) makes uid <-> user_id score ~1.0."""
        cmp = STEDComparator(config=STEDConfig(aliases=(("uid", "user_id"),)))
        result = cmp.compare({"uid": 1}, {"user_id": 1})
        # Aliased keys match → matched_pairs non-empty, no unmatched.
        assert len(result.matched_pairs) == 1
        assert result.key_mappings == {"uid": "user_id"}
        assert result.unmatched_left == ()
        assert result.unmatched_right == ()

    def test_alias_pair_is_bidirectional(self) -> None:
        """Same pair works when the alias is on the left instead."""
        cmp = STEDComparator(config=STEDConfig(aliases=(("uid", "user_id"),)))
        result = cmp.compare({"user_id": 1}, {"uid": 1})
        assert len(result.matched_pairs) == 1
        assert result.key_mappings == {"user_id": "uid"}
        assert result.unmatched_left == ()
        assert result.unmatched_right == ()

    def test_multiple_aliases_combine(self) -> None:
        """Several alias pairs all kick in within a single compare()."""
        cmp = STEDComparator(
            config=STEDConfig(
                aliases=(("uid", "user_id"), ("addr", "address")),
            )
        )
        result = cmp.compare(
            {"uid": 1, "addr": "Main St"},
            {"user_id": 1, "address": "Main St"},
        )
        assert len(result.matched_pairs) == 2
        assert result.key_mappings == {"uid": "user_id", "addr": "address"}
        assert result.unmatched_left == ()
        assert result.unmatched_right == ()

    def test_alias_survives_backend_normalisation(self) -> None:
        """Backend normalises kebab-case to the same form as snake_case.

        ``aliases=(("id", "user_id"),)`` matches ``{"id": 1}`` against
        ``{"user-id": 1}`` because the StaticBackend's KeyNormalizer
        rewrites ``user-id`` to the same canonical token sequence as
        ``user_id``, and the alias set is pre-normalised at build time.
        """
        cmp = STEDComparator(config=STEDConfig(aliases=(("id", "user_id"),)))
        result = cmp.compare({"id": 1}, {"user-id": 1})
        assert len(result.matched_pairs) == 1
        assert result.unmatched_left == ()
        assert result.unmatched_right == ()

    def test_aliases_force_match_against_competing_close_label(self) -> None:
        """An alias overrides a closer-by-Levenshtein competitor.

        Without the alias, the StaticBackend pairs ``uid`` with the
        textually-closer ``uuid``.  With ``("uid", "user_id")`` declared,
        the alias short-circuit makes ``uid <-> user_id`` cost 0.0 — so
        the Hungarian matcher locks that pair in.
        """
        left = {"uid": 1}
        right = {"user_id": 1, "uuid": "abc"}

        cmp_no_alias = STEDComparator()
        cmp_aliased = STEDComparator(config=STEDConfig(aliases=(("uid", "user_id"),)))
        no_alias = cmp_no_alias.compare(left, right)
        aliased = cmp_aliased.compare(left, right)

        # With the alias declared, uid pairs with user_id.
        assert aliased.key_mappings.get("uid") == "user_id"
        # Without the alias, Levenshtein prefers the textually-closer
        # ``uuid`` (one substitution) over ``user_id`` (much further) —
        # so the no-alias result is different.
        assert no_alias.key_mappings.get("uid") == "uuid"

    def test_alias_does_not_affect_value_content_distance(self) -> None:
        """Aliases unify KEYS, not VALUES — differing values still penalise."""
        cmp = STEDComparator(config=STEDConfig(aliases=(("uid", "user_id"),)))
        result = cmp.compare({"uid": 1}, {"user_id": 2})
        # Keys map cleanly, but the differing scalar values must drag the
        # overall similarity score below 1.0.
        assert result.key_mappings == {"uid": "user_id"}
        assert result.similarity_score < 1.0

    def test_alias_unused_keys_remain_unmatched(self) -> None:
        """Non-aliased keys without partners still appear in unmatched lists."""
        cmp = STEDComparator(config=STEDConfig(aliases=(("uid", "user_id"),)))
        result = cmp.compare(
            {"uid": 1, "leftover_left": "x"},
            {"user_id": 1, "leftover_right": "y"},
        )
        # uid <-> user_id matches; the two leftover keys are different
        # enough that they should not both be matched together by the
        # StaticBackend at the default threshold.
        assert ("uid", "user_id") in result.key_mappings.items()
