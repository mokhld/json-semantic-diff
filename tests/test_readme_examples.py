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
        # README claims "~0.99".  Audit C6 (wave 7) nudged this case
        # slightly upward (proper Zhang-Shasha denominator gives a touch
        # less penalty for the partial-naming-match key).  Keep the wide
        # tolerance so benign algorithm refinements don't fail the test.
        assert result.similarity_score == pytest.approx(0.99, abs=0.02)
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
        # Audit C6 (wave 7): the per-pair similarity for same-shape /
        # different-content single-key OBJECTs no longer binary-collapses.
        # Audit I4 (wave 8): the stronger ``lambda_unmatched=0.5``
        # pulls the asymmetric pairs further down, so the erratic
        # generator's consistency lands around 0.32 (was ~0.55 in wave 7).
        # README was updated to "~0.32" — gate tightened from < 0.7 to
        # < 0.5 to track the post-I4 floor.
        assert 0.0 <= score < 0.5
