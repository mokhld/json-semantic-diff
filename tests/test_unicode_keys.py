"""Unicode key handling (audit gap T3).

These tests cover the M10/M11/M12 wave-1 fixes from a Unicode angle: keys
containing CJK characters, Latin-1 supplements, combining diacritics, and
emoji must all round-trip cleanly through ``compare``.  Path strings in
``unmatched_left`` / ``unmatched_right`` must remain valid ``str``
instances with no encoding surprises.
"""

from __future__ import annotations

from json_semantic_diff import compare


def test_cjk_key_identical_scores_one() -> None:
    """A CJK key compared to itself must score 1.0."""
    result = compare({"用户名": "Alice"}, {"用户名": "Alice"})
    assert result.similarity_score == 1.0


def test_combining_diacritic_key_identical_scores_one() -> None:
    """Latin-1 supplement / combining-diacritic keys are stable."""
    result = compare({"naïveKey": "x"}, {"naïveKey": "x"})
    assert result.similarity_score == 1.0


def test_emoji_key_identical_scores_one() -> None:
    """Astral-plane emoji in a key compares cleanly."""
    result = compare({"emoji_🚀": "x"}, {"emoji_🚀": "x"})
    assert result.similarity_score == 1.0


def test_accented_vs_unaccented_key_pin() -> None:
    """``café`` vs ``cafe`` keys: pin the *actual* observed behaviour.

    The library treats ``café`` and ``cafe`` as related-but-distinct keys
    — the key-matching layer detects the high textual similarity and
    pairs them, but the resulting similarity is less than 1.0.

    Audit C6 (wave 7): the score moved from 0.875 to 0.9375 because the
    Zhang-Shasha denominator (KEY + SCALAR = 2) gives the small remaining
    label cost half the weight it carried under the old ``len(children)``
    denominator.  Pinning the new value so any further change is an
    intentional decision.  Note that NFC/NFKC normalisation is *not*
    applied; accented and non-accented spellings remain distinct identities.
    """
    result = compare({"café": 1}, {"cafe": 1})
    assert 0.0 < result.similarity_score < 1.0
    assert result.similarity_score == 0.9375


def test_mixed_unicode_document_identical_scores_one() -> None:
    """A document combining CJK, accents, emoji, and nested accents."""
    payload = {
        "用户名": "Alice",
        "naïveKey": "x",
        "emoji_🚀": "rocket",
        "nested": {"café": 1, "menu": ["☕", "🥐", "🍰"]},
    }
    result = compare(payload, payload)
    assert result.similarity_score == 1.0


def test_unmatched_left_contains_valid_unicode_paths() -> None:
    """Paths in ``unmatched_left`` carrying Unicode keys must be valid ``str``.

    No surrogate escapes, no encoding errors, and round-tripping through
    UTF-8 must succeed.  This guards against accidental ``repr()``-based
    path-building that would mangle non-ASCII characters.
    """
    left = {"用户名": "Alice", "extra_用户": "foo", "emoji_🚀": "rocket"}
    right = {"用户名": "Alice"}
    result = compare(left, right)

    assert result.unmatched_left, "expected unmatched paths on the left"
    for path in result.unmatched_left:
        assert isinstance(path, str)
        # round-trips cleanly through UTF-8 — no lone surrogates etc.
        path.encode("utf-8").decode("utf-8")
        # path is not just a Python repr of the key — actual characters,
        # not ``\uXXXX`` escapes.
        assert "\\u" not in path


def test_unmatched_right_contains_valid_unicode_paths() -> None:
    """Mirror of the left-side test, exercising ``unmatched_right``."""
    left = {"用户名": "Alice"}
    right = {"用户名": "Alice", "extra_用户": "foo", "emoji_🚀": "rocket"}
    result = compare(left, right)

    assert result.unmatched_right, "expected unmatched paths on the right"
    for path in result.unmatched_right:
        assert isinstance(path, str)
        path.encode("utf-8").decode("utf-8")
        assert "\\u" not in path
