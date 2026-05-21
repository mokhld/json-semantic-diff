"""Tests for KeyNormalizer across all four naming conventions and edge cases.

Verifies all ROADMAP Phase 2 success criteria #3 normalization requirements:
- camelCase, PascalCase, snake_case, kebab-case
- Acronyms (APIKey, URLParser, URL)
- Digit boundaries (address2, v2Config)
- Mixed separators, multiple separators, edge cases
"""

import pytest

from json_semantic_diff.tree.normalizer import KeyNormalizer


@pytest.fixture
def normalizer() -> KeyNormalizer:
    """Provide a shared KeyNormalizer instance for all tests."""
    return KeyNormalizer()


class TestFourNamingConventions:
    """Test the four required naming conventions from ROADMAP success criteria."""

    def test_camel_case(self, normalizer: KeyNormalizer) -> None:
        """camelCase must split at lowercase->uppercase boundary."""
        assert normalizer.normalize("camelCase") == "camel case"

    def test_pascal_case(self, normalizer: KeyNormalizer) -> None:
        """PascalCase must split at uppercase-followed-by-lowercase boundary."""
        assert normalizer.normalize("PascalCase") == "pascal case"

    def test_kebab_case(self, normalizer: KeyNormalizer) -> None:
        """kebab-case hyphens must be replaced with spaces."""
        assert normalizer.normalize("kebab-case") == "kebab case"

    def test_snake_case(self, normalizer: KeyNormalizer) -> None:
        """snake_case underscores must be replaced with spaces."""
        assert normalizer.normalize("snake_case") == "snake case"


class TestAcronymHandling:
    """Test acronym normalization edge cases."""

    def test_acronym_prefix(self, normalizer: KeyNormalizer) -> None:
        """APIKey: acronym before a word -> 'api key'."""
        assert normalizer.normalize("APIKey") == "api key"

    def test_acronym_before_word(self, normalizer: KeyNormalizer) -> None:
        """URLParser: acronym immediately before capitalized word -> 'url parser'."""
        assert normalizer.normalize("URLParser") == "url parser"

    def test_all_caps(self, normalizer: KeyNormalizer) -> None:
        """All-caps acronym with no word following -> lowercased single word."""
        assert normalizer.normalize("URL") == "url"

    def test_acronym_in_middle(self, normalizer: KeyNormalizer) -> None:
        """parseHTMLContent -> 'parse html content'."""
        assert normalizer.normalize("parseHTMLContent") == "parse html content"


class TestDigitBoundaries:
    """Test letter/digit boundary normalization."""

    def test_trailing_digit(self, normalizer: KeyNormalizer) -> None:
        """address2: letter followed by digit inserts space."""
        assert normalizer.normalize("address2") == "address 2"

    def test_digit_in_version_prefix(self, normalizer: KeyNormalizer) -> None:
        """v2Config: leading version prefix splits digit and following word."""
        assert normalizer.normalize("v2Config") == "v 2 config"

    def test_digit_followed_by_letter(self, normalizer: KeyNormalizer) -> None:
        """2ndAddress: digit followed by letter inserts space."""
        assert normalizer.normalize("2ndAddress") == "2 nd address"

    def test_embedded_version(self, normalizer: KeyNormalizer) -> None:
        """api3Endpoint: digit boundaries on both sides."""
        assert normalizer.normalize("api3Endpoint") == "api 3 endpoint"


class TestMixedSeparators:
    """Test inputs combining multiple naming conventions or separators."""

    def test_mixed_underscore_and_hyphen(self, normalizer: KeyNormalizer) -> None:
        """Mixed separators: underscore and hyphen both converted to spaces."""
        assert normalizer.normalize("user_name-field") == "user name field"

    def test_multiple_underscores(self, normalizer: KeyNormalizer) -> None:
        """Multiple consecutive underscores collapse to a single space."""
        assert normalizer.normalize("some__key") == "some key"

    def test_multiple_hyphens(self, normalizer: KeyNormalizer) -> None:
        """Multiple consecutive hyphens collapse to a single space."""
        assert normalizer.normalize("some--key") == "some key"

    def test_underscore_with_camel(self, normalizer: KeyNormalizer) -> None:
        """camelCase parts within a snake_case key."""
        assert normalizer.normalize("first_lastName") == "first last name"


class TestEdgeCases:
    """Test boundary and degenerate inputs."""

    def test_empty_string(self, normalizer: KeyNormalizer) -> None:
        """Empty string normalizes to empty string."""
        assert normalizer.normalize("") == ""

    def test_single_word_lowercase(self, normalizer: KeyNormalizer) -> None:
        """Single lowercase word is returned unchanged."""
        assert normalizer.normalize("name") == "name"

    def test_already_normalized(self, normalizer: KeyNormalizer) -> None:
        """Pre-spaced input is returned as-is (whitespace collapses)."""
        assert normalizer.normalize("user name") == "user name"

    def test_leading_trailing_separators(self, normalizer: KeyNormalizer) -> None:
        """Leading/trailing underscores or hyphens produce no extra spaces."""
        assert normalizer.normalize("_private") == "private"
        assert normalizer.normalize("key_") == "key"

    def test_single_uppercase_letter(self, normalizer: KeyNormalizer) -> None:
        """Single uppercase letter is lowercased."""
        assert normalizer.normalize("A") == "a"

    def test_repeated_camel_words(self, normalizer: KeyNormalizer) -> None:
        """Multiple camelCase boundaries."""
        assert normalizer.normalize("fooBarBaz") == "foo bar baz"

    def test_output_is_always_lowercase(self, normalizer: KeyNormalizer) -> None:
        """normalize() output is always lowercase regardless of input."""
        result = normalizer.normalize("MixedCASEInput")
        assert result == result.lower()

    def test_whitespace_is_collapsed(self, normalizer: KeyNormalizer) -> None:
        """Internal whitespace in output is always single spaces."""
        result = normalizer.normalize("some___key")
        assert "  " not in result

    def test_no_leading_or_trailing_whitespace(self, normalizer: KeyNormalizer) -> None:
        """normalize() output has no leading or trailing whitespace."""
        result = normalizer.normalize("  spaced  ")
        # Spaces treated as whitespace; split() collapses them
        assert result == result.strip()


# ---------------------------------------------------------------------------
# Unicode handling (M10): non-ASCII keys must NOT get spurious case splits
# ---------------------------------------------------------------------------


class TestUnicodeKeys:
    """Unicode keys are treated as a single indivisible token.

    Policy: case-class regexes use ``re.ASCII`` semantics, so scripts
    without an upper/lower distinction (CJK, etc.) and fully-lowercase
    precomposed Latin keys (e.g. naïveKey) preserve the SAME splitting
    behaviour for the ASCII portions; the non-ASCII portion never injects
    a spurious boundary.
    """

    def test_cjk_key_is_single_token(self, normalizer: KeyNormalizer) -> None:
        """CJK characters (no case) -> returned unchanged, no spaces added."""
        result = normalizer.normalize("用户名")
        assert result == "用户名"

    def test_naive_key_no_spurious_split(self, normalizer: KeyNormalizer) -> None:
        """'naïveKey' splits exactly at 'e|K' (ASCII boundary), producing
        'naïve key' — the ï/ASCII boundary must NOT introduce an extra split."""
        result = normalizer.normalize("naïveKey")
        # Lowercase final, only one ASCII camel boundary (e -> K)
        assert result == "naïve key"

    def test_pure_unicode_lowercase_passes_through(
        self, normalizer: KeyNormalizer
    ) -> None:
        """Pure non-ASCII lowercase word: unchanged (just lowercased + collapsed)."""
        result = normalizer.normalize("café")
        assert result == "café"

    def test_unicode_with_underscore_split(self, normalizer: KeyNormalizer) -> None:
        """Underscore separator still splits; unicode tokens preserved."""
        result = normalizer.normalize("用户_名")
        assert result == "用户 名"

    def test_eastern_arabic_digits_not_split(self, normalizer: KeyNormalizer) -> None:
        """Non-ASCII digits (Eastern Arabic ٢) must NOT trigger the
        digit-boundary rule — ASCII flag keeps ``\\d`` ASCII-only."""
        result = normalizer.normalize("address٢")
        # No split between 'address' and '٢' because ٢ is not an ASCII digit
        assert result == "address٢"
