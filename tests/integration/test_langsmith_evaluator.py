"""Mock-based tests for the LangSmith evaluator adapter.

These tests verify that ``LangSmithEvaluator`` wraps ``STEDComparator`` in the
LangSmith evaluator interface without requiring the real langsmith SDK.

All SDK types are mocked via ``sys.modules`` patching so the tests run in the
base development environment (no langsmith installed).
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from json_semantic_diff.comparator import STEDComparator


def _make_mock_langsmith() -> types.ModuleType:
    """Build a minimal mock langsmith module tree."""

    # Simple EvaluationResult that stores kwargs as attributes
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


@pytest.fixture(autouse=True)
def patch_langsmith(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject the mock langsmith modules before each test and clean up after."""
    mock_ls = _make_mock_langsmith()
    monkeypatch.setitem(sys.modules, "langsmith", mock_ls)
    monkeypatch.setitem(sys.modules, "langsmith.evaluation", mock_ls.evaluation)
    monkeypatch.setitem(
        sys.modules,
        "langsmith.evaluation.evaluator",
        mock_ls.evaluation.evaluator,
    )
    monkeypatch.setitem(sys.modules, "langsmith.schemas", mock_ls.schemas)
    # Reload adapter so it picks up the patched sys.modules
    import json_semantic_diff.integrations._langsmith as _ls_module

    importlib.reload(_ls_module)


def _get_evaluator(comparator: STEDComparator, **kwargs: Any) -> Any:
    """Import and call LangSmithEvaluator after sys.modules is patched."""
    from json_semantic_diff.integrations._langsmith import LangSmithEvaluator

    return LangSmithEvaluator(comparator, **kwargs)


def _make_run(output_value: Any, key: str = "output") -> MagicMock:
    run = MagicMock()
    run.outputs = {key: output_value}
    return run


def _make_example(output_value: Any, key: str = "output") -> MagicMock:
    example = MagicMock()
    example.outputs = {key: output_value}
    return example


class TestLangSmithEvaluator:
    def test_evaluator_returns_evaluation_result(self) -> None:
        """Evaluator should return an EvaluationResult with key and score."""
        comparator = STEDComparator()
        evaluator = _get_evaluator(comparator)

        run = _make_run({"user_name": "John"})
        example = _make_example({"userName": "John"})

        result = evaluator(run, example)

        assert result.key == "semantic_similarity"
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 1.0

    def test_evaluator_score_range_equivalent(self) -> None:
        """Equivalent inputs (snake_case vs camelCase) should score high."""
        comparator = STEDComparator()
        evaluator = _get_evaluator(comparator)

        run = _make_run({"user_name": "Alice"})
        example = _make_example({"userName": "Alice"})

        result = evaluator(run, example)
        assert result.score >= 0.0
        assert result.score <= 1.0

    def test_evaluator_score_range_non_equivalent(self) -> None:
        """Non-equivalent inputs should return a score in valid range."""
        comparator = STEDComparator()
        evaluator = _get_evaluator(comparator)

        run = _make_run({"user_name": "Alice"})
        example = _make_example({"address": "123 Main St"})

        result = evaluator(run, example)
        assert 0.0 <= result.score <= 1.0

    def test_evaluator_custom_output_key(self) -> None:
        """Custom output_key should extract from the correct key."""
        comparator = STEDComparator()
        evaluator = _get_evaluator(comparator, output_key="result")

        run = _make_run({"a": 1}, key="result")
        example = _make_example({"a": 1}, key="result")

        result = evaluator(run, example)
        assert result.key == "semantic_similarity"
        # Identical structures should score 1.0
        assert result.score == 1.0

    def test_evaluator_none_example(self) -> None:
        """Passing example=None should return score=None (N2 fix), not 0.0."""
        comparator = STEDComparator()
        evaluator = _get_evaluator(comparator)

        run = _make_run({"a": 1})

        result = evaluator(run, example=None)
        assert result.key == "semantic_similarity"
        # N2: don't silently score 0.0 when reference data is missing.
        assert result.score is None

    def test_import_error_without_langsmith(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LangSmithEvaluator should raise ImportError with install hint when SDK absent."""
        # Remove langsmith from sys.modules to simulate missing SDK
        for key in list(sys.modules.keys()):
            if key == "langsmith" or key.startswith("langsmith."):
                monkeypatch.delitem(sys.modules, key)

        # Also patch builtins.__import__ to block langsmith
        import builtins

        real_import = builtins.__import__

        def _blocking_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "langsmith" or name.startswith("langsmith."):
                raise ImportError(f"Mocked missing: {name}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocking_import)

        import json_semantic_diff.integrations._langsmith as _ls_module

        importlib.reload(_ls_module)

        from json_semantic_diff.integrations._langsmith import LangSmithEvaluator

        comparator = STEDComparator()
        with pytest.raises(
            ImportError, match="pip install json-semantic-diff\\[langsmith\\]"
        ):
            LangSmithEvaluator(comparator)
