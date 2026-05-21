"""Mock-based tests for the Braintrust scorer adapter.

These tests verify that ``BraintrustScorer`` wraps ``STEDComparator`` in the
Braintrust scorer interface.  No Braintrust SDK is required — the scorer is a
plain Python function.
"""

from __future__ import annotations

from typing import Any

import pytest

from json_semantic_diff.comparator import STEDComparator
from json_semantic_diff.integrations._braintrust import BraintrustScorer


class TestBraintrustScorer:
    @pytest.fixture
    def comparator(self) -> STEDComparator:
        return STEDComparator()

    @pytest.fixture
    def scorer(self, comparator: STEDComparator) -> Any:
        return BraintrustScorer(comparator)

    def test_scorer_returns_float(self, scorer: Any) -> None:
        """Scorer should return a float in [0.0, 1.0] when expected is provided."""
        result = scorer(
            input={},
            output={"user_name": "x"},
            expected={"userName": "x"},
        )
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_scorer_returns_none_without_expected(self, scorer: Any) -> None:
        """Scorer should return None when expected is not provided."""
        result = scorer(input={}, output={"a": 1})
        assert result is None

    def test_scorer_returns_none_with_explicit_none_expected(self, scorer: Any) -> None:
        """Scorer should return None when expected=None is passed explicitly."""
        result = scorer(input={}, output={"a": 1}, expected=None)
        assert result is None

    def test_scorer_name(self, scorer: Any) -> None:
        """Scorer __name__ should be 'semantic_similarity' for Braintrust display."""
        assert scorer.__name__ == "semantic_similarity"

    def test_scorer_structural_break(self, scorer: Any) -> None:
        """Structurally different documents should score below the equivalence band.

        Audit C6 (wave 7): single-key OBJECTs with KEY -> SCALAR shape have a
        Zhang-Shasha floor around 0.5 — the matched-pair raw distance no
        longer binary-collapses against a ``len(children)=1`` denominator.
        """
        result = scorer(
            input={},
            output={"user_name": "x"},
            expected={"address": "123"},
        )
        assert isinstance(result, float)
        assert result < 0.6

    def test_scorer_identical(self, scorer: Any) -> None:
        """Identical output and expected should score 1.0."""
        payload = {"user_name": "Alice", "age": 30}
        result = scorer(input={}, output=payload, expected=payload)
        assert result == pytest.approx(1.0)

    def test_scorer_metadata_ignored(self, scorer: Any) -> None:
        """metadata parameter should be accepted but not affect the score."""
        result_without = scorer(
            input={},
            output={"a": 1},
            expected={"a": 1},
        )
        result_with = scorer(
            input={},
            output={"a": 1},
            expected={"a": 1},
            metadata={"extra": "context"},
        )
        assert result_without == pytest.approx(result_with)
