"""Integration tests for F15: eval-adapter metadata enrichment.

Each adapter now surfaces ``key_mappings``, ``unmatched_left``,
``unmatched_right``, and ``computation_time_ms`` from the underlying
``ComparisonResult`` so the data is visible in the eval platform UI.

These tests also cover the N2/N3 audit fixes — when reference data is
missing, adapters return a score of ``None`` (or platform equivalent)
rather than silently scoring ``0.0``.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from json_semantic_diff.comparator import STEDComparator

# ---------------------------------------------------------------------------
# LangSmith adapter
# ---------------------------------------------------------------------------


def _make_mock_langsmith() -> types.ModuleType:
    class EvaluationResult:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    mock_langsmith = MagicMock()
    mock_langsmith.evaluation = MagicMock()
    mock_langsmith.evaluation.evaluator = MagicMock()
    mock_langsmith.evaluation.evaluator.EvaluationResult = EvaluationResult
    mock_langsmith.schemas = MagicMock()
    mock_langsmith.schemas.Run = MagicMock
    mock_langsmith.schemas.Example = MagicMock
    return mock_langsmith  # type: ignore[return-value]


class TestLangSmithMetadata:
    @pytest.fixture(autouse=True)
    def patch_langsmith(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_ls = _make_mock_langsmith()
        monkeypatch.setitem(sys.modules, "langsmith", mock_ls)
        monkeypatch.setitem(sys.modules, "langsmith.evaluation", mock_ls.evaluation)
        monkeypatch.setitem(
            sys.modules,
            "langsmith.evaluation.evaluator",
            mock_ls.evaluation.evaluator,
        )
        monkeypatch.setitem(sys.modules, "langsmith.schemas", mock_ls.schemas)
        import json_semantic_diff.integrations._langsmith as _ls_module

        importlib.reload(_ls_module)

    def _evaluator(self, **kwargs: Any) -> Any:
        from json_semantic_diff.integrations._langsmith import LangSmithEvaluator

        return LangSmithEvaluator(STEDComparator(), **kwargs)

    def _make_run(self, output_value: Any) -> MagicMock:
        run = MagicMock()
        run.outputs = {"output": output_value}
        return run

    def _make_example(self, output_value: Any) -> MagicMock:
        example = MagicMock()
        example.outputs = {"output": output_value}
        return example

    def test_comment_carries_audit_fields(self) -> None:
        """Comment is JSON with key_mappings, unmatched_*, and computation_time_ms."""
        evaluator = self._evaluator()
        run = self._make_run({"user_name": "Alice"})
        example = self._make_example({"userName": "Alice"})

        result = evaluator(run, example)

        assert result.score is not None
        assert hasattr(result, "comment")
        payload = json.loads(result.comment)
        assert "key_mappings" in payload
        assert "unmatched_left" in payload
        assert "unmatched_right" in payload
        assert "computation_time_ms" in payload
        assert isinstance(payload["key_mappings"], dict)
        assert isinstance(payload["unmatched_left"], list)
        assert isinstance(payload["unmatched_right"], list)
        assert isinstance(payload["computation_time_ms"], float)

    def test_comment_reflects_matched_keys(self) -> None:
        """Snake/camel case keys should appear in the key_mappings comment."""
        evaluator = self._evaluator()
        run = self._make_run({"user_name": "Alice"})
        example = self._make_example({"userName": "Alice"})

        result = evaluator(run, example)
        payload = json.loads(result.comment)
        assert payload["key_mappings"] == {"user_name": "userName"}

    def test_n2_run_outputs_none_returns_score_none(self) -> None:
        """Regression for N2: run.outputs = None → score=None, not 0.0."""
        evaluator = self._evaluator()
        run = MagicMock()
        run.outputs = None
        example = self._make_example({"a": 1})

        result = evaluator(run, example)
        assert result.score is None
        payload = json.loads(result.comment)
        assert payload["skipped"] is True

    def test_n2_run_outputs_missing_key_returns_score_none(self) -> None:
        """Regression for N2: missing output_key → score=None, not 0.0."""
        evaluator = self._evaluator()
        run = MagicMock()
        run.outputs = {"different_key": 1}
        example = self._make_example({"a": 1})

        result = evaluator(run, example)
        assert result.score is None

    def test_n2_example_none_returns_score_none(self) -> None:
        """example=None → score=None, not 0.0."""
        evaluator = self._evaluator()
        run = self._make_run({"a": 1})

        result = evaluator(run, example=None)
        assert result.score is None

    def test_n2_example_outputs_missing_key_returns_score_none(self) -> None:
        """example.outputs missing the key → score=None."""
        evaluator = self._evaluator()
        run = self._make_run({"a": 1})
        example = MagicMock()
        example.outputs = {"other_key": 1}

        result = evaluator(run, example)
        assert result.score is None


# ---------------------------------------------------------------------------
# Braintrust adapter
# ---------------------------------------------------------------------------


class TestBraintrustMetadata:
    def test_score_carries_metadata_attribute(self) -> None:
        """BraintrustScore subclass of float carries a `.metadata` dict."""
        from json_semantic_diff.integrations._braintrust import (
            BraintrustScore,
            BraintrustScorer,
        )

        scorer = BraintrustScorer(STEDComparator())
        result = scorer(
            input={},
            output={"user_name": "Alice"},
            expected={"userName": "Alice"},
        )

        # Existing return-type contract preserved.
        assert isinstance(result, float)
        # New: audit-trail metadata accessible as attribute.
        assert isinstance(result, BraintrustScore)
        assert isinstance(result.metadata, dict)
        for field in (
            "key_mappings",
            "unmatched_left",
            "unmatched_right",
            "computation_time_ms",
        ):
            assert field in result.metadata

    def test_metadata_reflects_matched_keys(self) -> None:
        """Snake/camel case keys appear in metadata.key_mappings."""
        from json_semantic_diff.integrations._braintrust import BraintrustScorer

        scorer = BraintrustScorer(STEDComparator())
        result = scorer(
            input={},
            output={"user_name": "Alice"},
            expected={"userName": "Alice"},
        )
        assert result is not None
        assert result.metadata["key_mappings"] == {"user_name": "userName"}

    def test_none_expected_still_returns_none(self) -> None:
        """expected=None must continue to return None (existing contract)."""
        from json_semantic_diff.integrations._braintrust import BraintrustScorer

        scorer = BraintrustScorer(STEDComparator())
        assert scorer(input={}, output={"a": 1}, expected=None) is None

    def test_float_arithmetic_still_works(self) -> None:
        """The float subclass remains usable in arithmetic operations."""
        from json_semantic_diff.integrations._braintrust import BraintrustScorer

        scorer = BraintrustScorer(STEDComparator())
        result = scorer(input={}, output={"a": 1}, expected={"a": 1})
        assert result is not None
        # Arithmetic compatibility.
        assert result + 0.0 == pytest.approx(1.0)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Weave adapter
# ---------------------------------------------------------------------------


def _make_mock_weave() -> types.ModuleType:
    class MockScorer:
        pass

    mock_weave = MagicMock()
    mock_weave.Scorer = MockScorer
    mock_weave.op = lambda fn: fn  # type: ignore[assignment]
    return mock_weave  # type: ignore[return-value]


class TestWeaveMetadata:
    @pytest.fixture(autouse=True)
    def patch_weave(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_w = _make_mock_weave()
        monkeypatch.setitem(sys.modules, "weave", mock_w)
        import json_semantic_diff.integrations._weave as _w_module

        importlib.reload(_w_module)

    def _scorer(self) -> Any:
        from json_semantic_diff.integrations._weave import WeaveScorer

        return WeaveScorer(STEDComparator())

    def test_score_dict_carries_audit_fields(self) -> None:
        """Score dict includes all audit-trail fields."""
        scorer = self._scorer()
        result = scorer.score(
            output={"user_name": "Alice"},
            target={"userName": "Alice"},
        )
        assert "semantic_similarity" in result
        assert "key_mappings" in result
        assert "unmatched_left" in result
        assert "unmatched_right" in result
        assert "computation_time_ms" in result

    def test_score_dict_reflects_matched_keys(self) -> None:
        """key_mappings entry mirrors the underlying ComparisonResult."""
        scorer = self._scorer()
        result = scorer.score(
            output={"user_name": "Alice"},
            target={"userName": "Alice"},
        )
        assert result["key_mappings"] == {"user_name": "userName"}

    def test_n3_none_target_returns_score_none(self) -> None:
        """Regression for N3: target=None → semantic_similarity=None, not 0.0."""
        scorer = self._scorer()
        result = scorer.score(output={"a": 1}, target=None)
        assert result["semantic_similarity"] is None
        assert result.get("skipped") is True
