"""Tests for json_semantic_diff.api.format_diff.

Covers the spec for F6 (human-readable diff renderer):

- Identical inputs render the header + a single Matched section.
- A wholly mismatched diff renders all three sections.
- Empty objects render header only (no item sections).
- ``indent`` controls leading whitespace on item lines.
- Output is deterministic across runs (sort by left label / by path).
- Snapshot of a small canonical diff.
"""

from __future__ import annotations

import pytest

from json_semantic_diff import ComparisonResult, compare, format_diff

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    similarity: float = 0.5,
    matched: tuple[tuple[str, str], ...] = (),
    key_mappings: dict[str, str] | None = None,
    unmatched_left: tuple[str, ...] = (),
    unmatched_right: tuple[str, ...] = (),
    ms: float = 1.0,
) -> ComparisonResult:
    """Build a fully-specified ComparisonResult for snapshot testing."""
    return ComparisonResult(
        similarity_score=similarity,
        matched_pairs=matched,
        key_mappings=dict(key_mappings or {}),
        unmatched_left=unmatched_left,
        unmatched_right=unmatched_right,
        computation_time_ms=ms,
    )


# ---------------------------------------------------------------------------
# Header line
# ---------------------------------------------------------------------------


class TestHeaderLine:
    def test_header_present_for_identical(self) -> None:
        out = format_diff(compare({"a": 1}, {"a": 1}))
        assert out.startswith("Similarity: 1.00")
        assert "computed in" in out
        assert "ms" in out

    def test_header_for_empty_inputs(self) -> None:
        out = format_diff(compare({}, {}))
        # Empty objects compare as identical, but have no KEY children
        # → no Matched section.  Output is header only.
        assert out.startswith("Similarity: 1.00")
        assert "Matched" not in out
        assert "Unmatched" not in out

    def test_header_formatted_to_two_decimals(self) -> None:
        out = format_diff(_make_result(similarity=0.87654))
        # 0.87654 -> 0.88 (banker's rounding by format spec)
        first_line = out.splitlines()[0]
        assert "Similarity: 0.88" in first_line

    def test_header_timing_formatted_to_one_decimal(self) -> None:
        out = format_diff(_make_result(ms=1.347))
        first_line = out.splitlines()[0]
        assert "1.3 ms" in first_line


# ---------------------------------------------------------------------------
# Section gating
# ---------------------------------------------------------------------------


class TestSectionGating:
    def test_identical_inputs_produce_matched_section_only(self) -> None:
        out = format_diff(compare({"a": 1, "b": 2}, {"a": 1, "b": 2}))
        assert "Matched (2):" in out
        assert "Unmatched in left" not in out
        assert "Unmatched in right" not in out

    def test_wholly_mismatched_produces_all_three_sections(self) -> None:
        result = _make_result(
            matched=(("/x", "/y"),),
            unmatched_left=("/timestamp",),
            unmatched_right=("/createdAt",),
        )
        out = format_diff(result)
        assert "Matched (1):" in out
        assert "Unmatched in left (1):" in out
        assert "Unmatched in right (1):" in out

    def test_empty_matched_section_skipped(self) -> None:
        result = _make_result(
            matched=(),
            unmatched_left=("/a",),
            unmatched_right=("/b",),
        )
        out = format_diff(result)
        assert "Matched" not in out

    def test_empty_unmatched_left_section_skipped(self) -> None:
        result = _make_result(
            matched=(("/x", "/x"),),
            unmatched_left=(),
            unmatched_right=("/b",),
        )
        out = format_diff(result)
        assert "Unmatched in left" not in out

    def test_empty_unmatched_right_section_skipped(self) -> None:
        result = _make_result(
            matched=(("/x", "/x"),),
            unmatched_left=("/a",),
            unmatched_right=(),
        )
        out = format_diff(result)
        assert "Unmatched in right" not in out


# ---------------------------------------------------------------------------
# Indent parameter
# ---------------------------------------------------------------------------


class TestIndentParameter:
    def test_default_indent_is_two_spaces(self) -> None:
        out = format_diff(compare({"a": 1}, {"a": 1}))
        # Find an item line (after the Matched header)
        for line in out.splitlines():
            if "<->" in line:
                assert line.startswith("  ")
                assert not line.startswith("   ")
                break
        else:
            raise AssertionError("no matched item line found")

    def test_indent_four_renders_four_spaces(self) -> None:
        out = format_diff(compare({"a": 1}, {"a": 1}), indent=4)
        for line in out.splitlines():
            if "<->" in line:
                assert line.startswith("    ")
                break
        else:
            raise AssertionError("no matched item line found")

    def test_indent_zero_renders_no_leading_space(self) -> None:
        out = format_diff(compare({"a": 1}, {"a": 1}), indent=0)
        for line in out.splitlines():
            if "<->" in line:
                assert not line.startswith(" ")
                break
        else:
            raise AssertionError("no matched item line found")

    def test_negative_indent_clamps_to_zero(self) -> None:
        # Treated as 0 — no exception.
        out = format_diff(compare({"a": 1}, {"a": 1}), indent=-3)
        for line in out.splitlines():
            if "<->" in line:
                assert not line.startswith(" ")
                break

    def test_indent_non_int_raises(self) -> None:
        with pytest.raises(TypeError, match="indent must be an int"):
            format_diff(compare({"a": 1}, {"a": 1}), indent="two")  # type: ignore[arg-type]

    def test_indent_bool_rejected(self) -> None:
        # bool is a subclass of int but is not a sensible indent value.
        with pytest.raises(TypeError, match="indent must be an int"):
            format_diff(compare({"a": 1}, {"a": 1}), indent=True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_matched_pairs_sorted_by_left_label(self) -> None:
        result = _make_result(
            matched=(
                ("/zebra", "/zebra"),
                ("/apple", "/apple"),
                ("/mango", "/mango"),
            ),
        )
        out = format_diff(result)
        # Body lines containing <->
        body = [ln for ln in out.splitlines() if "<->" in ln]
        labels = [ln.strip().split(" ", 1)[0] for ln in body]
        assert labels == sorted(labels) == ["apple", "mango", "zebra"]

    def test_unmatched_left_sorted_alphabetically(self) -> None:
        result = _make_result(
            unmatched_left=("/zebra", "/apple", "/mango"),
        )
        out = format_diff(result)
        body = [ln.strip() for ln in out.splitlines() if ln.startswith(" ")]
        assert body == ["/apple", "/mango", "/zebra"]

    def test_unmatched_right_sorted_alphabetically(self) -> None:
        result = _make_result(
            unmatched_right=("/zebra", "/apple"),
        )
        out = format_diff(result)
        body = [ln.strip() for ln in out.splitlines() if ln.startswith(" ")]
        assert body == ["/apple", "/zebra"]

    def test_output_stable_across_repeated_renders(self) -> None:
        # Re-render the same result twice — must match byte for byte.
        result = _make_result(
            matched=(("/b", "/b"), ("/a", "/a")),
            unmatched_left=("/c",),
            unmatched_right=("/d",),
        )
        assert format_diff(result) == format_diff(result)

    def test_output_stable_across_input_order(self) -> None:
        # Same content, different insertion order — output must agree
        # on the body sections (header timing line will differ).
        r1 = _make_result(
            matched=(("/a", "/a"), ("/b", "/b")),
            unmatched_left=("/x", "/y"),
        )
        r2 = _make_result(
            matched=(("/b", "/b"), ("/a", "/a")),
            unmatched_left=("/y", "/x"),
        )
        body1 = "\n".join(format_diff(r1).splitlines()[1:])
        body2 = "\n".join(format_diff(r2).splitlines()[1:])
        assert body1 == body2


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_snapshot_value_mismatch(self) -> None:
        """compare({"a": 1}, {"a": 2}) produces a sensible output."""
        out = format_diff(compare({"a": 1}, {"a": 2}))
        # Cannot snapshot exact similarity (depends on weights) or timing,
        # but we can pin the *shape*.
        lines = out.splitlines()
        assert lines[0].startswith("Similarity: ")
        assert "(computed in " in lines[0]
        # Same key on both sides → matched, no unmatched
        assert any("Matched (1):" in ln for ln in lines)
        assert any("a <-> a" in ln for ln in lines)
        assert "Unmatched in left" not in out
        assert "Unmatched in right" not in out

    def test_snapshot_three_section_diff(self) -> None:
        result = _make_result(
            similarity=0.5,
            matched=(("/email", "/emailAddress"),),
            unmatched_left=("/legacy",),
            unmatched_right=("/createdAt",),
            ms=1.5,
        )
        expected = (
            "Similarity: 0.50  (computed in 1.5 ms)\n"
            "\n"
            "Matched (1):\n"
            "  email <-> emailAddress\n"
            "\n"
            "Unmatched in left (1):\n"
            "  /legacy\n"
            "\n"
            "Unmatched in right (1):\n"
            "  /createdAt"
        )
        assert format_diff(result) == expected
