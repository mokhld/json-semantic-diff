"""Pin README code examples to their actual numeric output.

If the algorithm changes, these tests will fail loudly, forcing a coordinated
README + code update. Keep this file in lock-step with README.md's "Quick Start"
section.
"""

from __future__ import annotations

import pytest

from json_semantic_diff import (
    compare,
    consistency_score,
    is_equivalent,
    similarity_score,
)


class TestCompareExample:
    """README "Compare two JSON documents" section."""

    def test_fuzzy_key_match_scores_just_under_one(self) -> None:
        result = compare(
            {"user_name": "Alice", "email_address": "alice@corp.com"},
            {"userName": "Alice", "mailAddress": "alice@corp.com"},
        )
        # README claims "~0.98".  Use README's precision (abs=0.01) so a
        # benign algorithm refinement that still satisfies the documented
        # contract does not fail the test (test_readme review finding T3).
        assert result.similarity_score == pytest.approx(0.98, abs=0.01)
        assert result.key_mappings == {
            "user_name": "userName",
            "email_address": "mailAddress",
        }
        assert len(result.unmatched_left) == 0
        assert len(result.unmatched_right) == 0
        assert result.computation_time_ms > 0.0

    def test_naming_convention_only_is_identical(self) -> None:
        """README sub-example: pure naming-convention differences score 1.0."""
        result = compare(
            {"user_name": "Alice"},
            {"userName": "Alice"},
        )
        assert result.similarity_score == 1.0


class TestSimilarityScoreExample:
    """README "Quick similarity score" section."""

    def test_first_last_name_naming_variants(self) -> None:
        score = similarity_score(
            {"first_name": "Bob", "last_name": "Smith"},
            {"firstName": "Bob", "lastName": "Smith"},
        )
        # README claims 1.0 — pin it.
        assert score == 1.0


class TestIsEquivalentExample:
    """README "Boolean equivalence check" section."""

    def test_naming_difference_is_equivalent(self) -> None:
        assert is_equivalent({"user_name": "Alice"}, {"userName": "Alice"}) is True

    def test_different_structure_not_equivalent(self) -> None:
        assert is_equivalent({"name": "Alice"}, {"product": "Widget"}) is False

    def test_custom_threshold(self) -> None:
        # README shows a 0.99 threshold passing for a 1.0-score pair.
        assert (
            is_equivalent(
                {"user_name": "Alice"},
                {"userName": "Alice"},
                threshold=0.99,
            )
            is True
        )


class TestConsistencyScoreExample:
    """README "Measure generator consistency" section."""

    def test_stable_generator_scores_one(self) -> None:
        docs = [
            {"name": "Alice", "age": 30},
            {"name": "Alice", "age": 30},
            {"name": "Alice", "age": 30},
        ]
        assert consistency_score(docs) == 1.0

    def test_erratic_generator_scores_low(self) -> None:
        erratic = [
            {"name": "Alice", "age": 30},
            {"fullName": "Alice", "years": 30},
            {"person": "Alice"},
        ]
        score = consistency_score(erratic)
        # README claims ~0.15; allow a generous band so minor algorithm
        # tweaks don't break the test, but catch big regressions.
        assert 0.0 <= score < 0.4
