"""Mock-based tests for the W&B Weave scorer adapter.

These tests verify that ``WeaveScorer`` wraps ``STEDComparator`` in the
W&B Weave scorer interface without requiring the real weave SDK.

The weave SDK is mocked via ``sys.modules`` patching so the tests run in the
base development environment (no weave installed).
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from json_semantic_diff.comparator import STEDComparator


def _make_mock_weave() -> types.ModuleType:
    """Build a minimal mock weave module with a Scorer base class and op decorator."""

    class MockScorer:
        """Minimal Scorer base class — real weave.Scorer uses Pydantic."""

        pass

    mock_weave = MagicMock()
    mock_weave.Scorer = MockScorer
    # @weave.op is used as a decorator — use identity decorator in mock
    mock_weave.op = lambda fn: fn  # type: ignore[assignment]
    return mock_weave  # type: ignore[return-value]


@pytest.fixture(autouse=True)
def patch_weave(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject the mock weave modules before each test and clean up after."""
    mock_w = _make_mock_weave()
    monkeypatch.setitem(sys.modules, "weave", mock_w)
    # Reload adapter so it picks up the patched sys.modules
    import json_semantic_diff.integrations._weave as _weave_module

    importlib.reload(_weave_module)


def _get_scorer(comparator: STEDComparator) -> Any:
    """Import and call WeaveScorer after sys.modules is patched."""
    from json_semantic_diff.integrations._weave import WeaveScorer

    return WeaveScorer(comparator)


class TestWeaveScorer:
    def test_scorer_returns_dict(self) -> None:
        """Scorer.score() should return a dict with 'semantic_similarity' key."""
        comparator = STEDComparator()
        scorer = _get_scorer(comparator)

        result = scorer.score(
            output={"user_name": "x"},
            target={"userName": "x"},
        )

        assert isinstance(result, dict)
        assert "semantic_similarity" in result
        assert isinstance(result["semantic_similarity"], float)

    def test_scorer_none_target(self) -> None:
        """score() with target=None should return semantic_similarity=None (N3 fix).

        We previously silently returned 0.0 — that lied about the score on
        missing reference data.  The adapter now returns ``None`` plus a
        ``skipped`` flag.
        """
        comparator = STEDComparator()
        scorer = _get_scorer(comparator)

        result = scorer.score(output={"a": 1}, target=None)

        assert result["semantic_similarity"] is None
        assert result.get("skipped") is True

    def test_scorer_score_range(self) -> None:
        """All score values should be in [0.0, 1.0]."""
        comparator = STEDComparator()
        scorer = _get_scorer(comparator)

        # Equivalent (high score)
        r1 = scorer.score(
            output={"user_name": "Alice"},
            target={"userName": "Alice"},
        )
        assert 0.0 <= r1["semantic_similarity"] <= 1.0

        # Non-equivalent (low score, still valid range)
        r2 = scorer.score(
            output={"user_name": "Alice"},
            target={"address": "123 Main St"},
        )
        assert 0.0 <= r2["semantic_similarity"] <= 1.0

    def test_scorer_identical(self) -> None:
        """Identical output and target should score 1.0."""
        comparator = STEDComparator()
        scorer = _get_scorer(comparator)

        payload = {"user_name": "Alice", "age": 30}
        result = scorer.score(output=payload, target=payload)

        assert result["semantic_similarity"] == pytest.approx(1.0)

    def test_import_error_without_weave(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """WeaveScorer should raise ImportError with install hint when SDK absent."""
        # Remove weave from sys.modules to simulate missing SDK
        for key in list(sys.modules.keys()):
            if key == "weave" or key.startswith("weave."):
                monkeypatch.delitem(sys.modules, key)

        # Patch builtins.__import__ to block weave imports
        import builtins

        real_import = builtins.__import__

        def _blocking_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "weave" or name.startswith("weave."):
                raise ImportError(f"Mocked missing: {name}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocking_import)

        import json_semantic_diff.integrations._weave as _weave_module

        importlib.reload(_weave_module)

        from json_semantic_diff.integrations._weave import WeaveScorer

        comparator = STEDComparator()
        with pytest.raises(
            ImportError, match="pip install json-semantic-diff\\[weave\\]"
        ):
            WeaveScorer(comparator)
