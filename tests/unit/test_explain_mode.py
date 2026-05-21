"""Unit tests for explain mode (F1) on the public API.

These tests exercise ``ComparisonResult.explanation`` through the
``compare()`` entry point — the surface most users see.  Algorithm-level
behaviour is covered in ``tests/algorithm/test_sted.py``.
"""

from __future__ import annotations

import json

import pytest

from json_semantic_diff import compare
from json_semantic_diff.algorithm.config import STEDConfig
from json_semantic_diff.result import ComparisonResult, NodeContribution


class TestExplainModeDefault:
    """By default, ``collect_explanation`` is off and ``explanation`` is
    the empty tuple — preserving backward compatibility with callers that
    consumed ``ComparisonResult`` before explain mode existed."""

    def test_default_explanation_is_empty_tuple(self) -> None:
        result = compare({"a": 1, "b": 2}, {"a": 1, "b": 3})
        assert result.explanation == ()

    def test_default_explanation_is_omitted_from_dict(self) -> None:
        """``to_dict()`` must skip the ``explanation`` key when empty so the
        serialised payload matches the pre-explain format byte-for-byte."""
        result = compare({"a": 1}, {"a": 1})
        d = result.to_dict()
        assert "explanation" not in d

    def test_default_explanation_is_omitted_from_json(self) -> None:
        result = compare({"a": 1}, {"a": 1})
        loaded = json.loads(result.to_json())
        assert "explanation" not in loaded


class TestExplainModeOn:
    """When ``collect_explanation=True``, contributions populate
    ``ComparisonResult.explanation`` and survive JSON round-trip."""

    def test_explanation_present_for_divergent_inputs(self) -> None:
        config = STEDConfig(collect_explanation=True)
        result = compare({"a": 1}, {"a": 2}, config=config)
        assert result.explanation, "expected non-empty explanation"
        assert all(isinstance(c, NodeContribution) for c in result.explanation)

    def test_explanation_empty_for_identical_inputs(self) -> None:
        config = STEDConfig(collect_explanation=True)
        result = compare({"a": 1, "b": "x"}, {"a": 1, "b": "x"}, config=config)
        assert result.explanation == ()

    def test_to_dict_includes_explanation_when_non_empty(self) -> None:
        config = STEDConfig(collect_explanation=True)
        result = compare({"a": 1, "extra": 2}, {"a": 9}, config=config)
        d = result.to_dict()
        assert "explanation" in d
        assert isinstance(d["explanation"], list)
        assert all(
            set(entry.keys()) == {"path", "contribution", "kind", "detail"}
            for entry in d["explanation"]
        )

    def test_to_json_round_trip_preserves_explanation(self) -> None:
        config = STEDConfig(collect_explanation=True)
        result = compare({"a": 1, "extra": 2}, {"a": 9}, config=config)
        loaded = json.loads(result.to_json())
        assert "explanation" in loaded
        assert len(loaded["explanation"]) == len(result.explanation)
        # First entry's contribution matches the (sorted descending) explanation.
        assert (
            loaded["explanation"][0]["contribution"]
            == result.explanation[0].contribution
        )

    def test_explanation_sorted_descending(self) -> None:
        config = STEDConfig(collect_explanation=True)
        left = {"a": "x", "b": "y", "extra1": 1, "extra2": 2, "extra3": 3}
        right = {"a": "X", "b": "Y", "different": "value"}
        result = compare(left, right, config=config)
        contributions = [c.contribution for c in result.explanation]
        assert contributions == sorted(contributions, reverse=True)

    def test_big_example_has_at_least_four_entries(self) -> None:
        """Spec: a document with 3 unmatched keys + 1 value mismatch must
        produce ≥ 4 contribution entries with sensible paths."""
        config = STEDConfig(collect_explanation=True)
        left = {
            "shared": "same",
            "value_changes": "old",
            "only_left_1": 1,
            "only_left_2": 2,
            "only_left_3": 3,
        }
        right = {
            "shared": "same",
            "value_changes": "new",
        }
        result = compare(left, right, config=config)
        # We expect 3 unmatched_left entries + 1 value_mismatch entry, minimum.
        assert len(result.explanation) >= 4
        kinds = [c.kind for c in result.explanation]
        assert kinds.count("unmatched_left") >= 3
        assert "value_mismatch" in kinds
        # Every path should be rooted with "/" and reference a real key.
        for c in result.explanation:
            assert c.path.startswith("/")
            assert c.path != ""
            assert c.contribution > 0.0

    def test_explanation_is_immutable_tuple(self) -> None:
        config = STEDConfig(collect_explanation=True)
        result = compare({"a": 1}, {"a": 2}, config=config)
        assert isinstance(result.explanation, tuple)

    def test_explanation_does_not_alter_similarity_score(self) -> None:
        """Sanity: turning explain mode on cannot perturb the score."""
        off = compare({"a": 1, "b": [1, 2, 3]}, {"a": 9, "b": [1, 2, 4]})
        on = compare(
            {"a": 1, "b": [1, 2, 3]},
            {"a": 9, "b": [1, 2, 4]},
            config=STEDConfig(collect_explanation=True),
        )
        assert on.similarity_score == pytest.approx(off.similarity_score)


class TestNodeContribution:
    """Direct tests on the NodeContribution data class."""

    def test_node_contribution_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        c = NodeContribution(path="/a", contribution=0.5, kind="matched")
        with pytest.raises(FrozenInstanceError):
            c.path = "/b"  # type: ignore[misc]

    def test_node_contribution_detail_defaults_to_empty(self) -> None:
        c = NodeContribution(path="/a", contribution=0.5, kind="matched")
        assert c.detail == ""

    def test_comparison_result_default_explanation_field(self) -> None:
        """``ComparisonResult`` can be constructed without ``explanation``."""
        r = ComparisonResult(
            similarity_score=1.0,
            matched_pairs=(),
            key_mappings={},
            unmatched_left=(),
            unmatched_right=(),
            computation_time_ms=0.0,
        )
        assert r.explanation == ()
