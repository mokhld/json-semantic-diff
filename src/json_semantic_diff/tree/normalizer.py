"""KeyNormalizer: converts JSON object keys to normalized lowercase words.

Handles four naming conventions:
- camelCase (e.g. "camelCase" -> "camel case")
- PascalCase (e.g. "PascalCase" -> "pascal case")
- snake_case (e.g. "snake_case" -> "snake case")
- kebab-case (e.g. "kebab-case" -> "kebab case")

Also handles:
- Acronyms (e.g. "APIKey" -> "api key", "URLParser" -> "url parser")
- Digit boundaries (e.g. "address2" -> "address 2", "v2Config" -> "v 2 config")
- Mixed separators (e.g. "user_name-field" -> "user name field")
- Multiple consecutive separators (e.g. "some__key" -> "some key")

Unicode policy: case-class regexes use ASCII semantics (``re.ASCII``) so
that scripts without an upper/lower distinction (e.g. CJK like "用户名",
fully-lower precomposed Latin like "naïve") are treated as a single
indivisible token.  The existing ASCII splitting behaviour is preserved
bit-for-bit; unicode keys simply do not gain spurious case boundaries.
"""

import re

# Compiled regex patterns (module-level, compiled once)

# Matches snake_case and kebab-case separators (underscores and hyphens)
_SEP = re.compile(r"[_\-]+")

# Matches camelCase boundary: lowercase ASCII letter followed by uppercase ASCII letter
# e.g. "camelCase" -> "camel Case" via "\1 \2"
# ASCII flag ensures non-Latin letters (e.g. "用户名") do NOT trigger splits.
_UPPER_LOWER = re.compile(r"([a-z])([A-Z])", re.ASCII)

# Matches acronym runs: sequence of uppercase letters before an uppercase+lowercase pair
# e.g. "URLParser" -> "URL Parser" via "\1 \2"
_UPPER_RUN = re.compile(r"([A-Z]+)([A-Z][a-z])", re.ASCII)

# Matches letter/digit boundaries in both directions
# e.g. "address2" -> "address 2", "v2Config" -> "v 2 Config"
# NOTE: Applied twice (passes 4a and 4b) because the regex engine consumes both
# characters in each match (e.g. "v2" in "v2Config"), so the other boundary
# ("2C") is not visible until a second pass after the first substitution.
# ASCII flag keeps `\d` ASCII-only (no Eastern Arabic numerals, etc.) and
# the [a-zA-Z] character class is already ASCII by construction.
_DIGIT_BOUNDARY = re.compile(r"([a-zA-Z])(\d)|(\d)([a-zA-Z])", re.ASCII)


class KeyNormalizer:
    """Normalizes JSON object keys to lowercase space-separated words.

    Uses a five-pass regex pipeline to handle camelCase, PascalCase,
    snake_case, kebab-case, acronyms, and digit boundaries.

    Example usage:
        normalizer = KeyNormalizer()
        normalizer.normalize("camelCase")    # "camel case"
        normalizer.normalize("APIKey")       # "api key"
        normalizer.normalize("snake_case")   # "snake case"
    """

    def normalize(self, key: str) -> str:
        """Normalize a JSON key to lowercase space-separated words.

        Processing pipeline (applied in order):
        1. Replace underscore and hyphen separators with spaces.
        2. Insert space at camelCase boundaries (lowercase->uppercase transitions).
        3. Insert space at acronym runs (e.g. "URL" before "Parser").
        4a. Insert space at digit boundaries (first pass — letter<->digit).
        4b. Insert space at digit boundaries (second pass — catches opposite side
            of isolated digits like the "C" in "v2Config" after "v2" is split).
        5. Lowercase everything and collapse whitespace.

        Args:
            key: The raw JSON object key string.

        Returns:
            Normalized lowercase space-separated string.
        """
        # Pass 1: replace separators with spaces
        s = _SEP.sub(" ", key)

        # Pass 2: insert space at camelCase boundary (xY -> x Y)
        s = _UPPER_LOWER.sub(r"\1 \2", s)

        # Pass 3: insert space at acronym run (URLParser -> URL Parser)
        s = _UPPER_RUN.sub(r"\1 \2", s)

        # Pass 4a: insert space at digit boundary — first pass (letter<->digit)
        s = _DIGIT_BOUNDARY.sub(r"\1\3 \2\4", s)

        # Pass 4b: second digit-boundary pass — handles the side of a digit that
        # was not reachable in 4a because the regex consumed both chars of the match
        # (e.g. "v2Config" -> "v 2Config" in 4a; "2C" boundary caught here -> "2 C")
        s = _DIGIT_BOUNDARY.sub(r"\1\3 \2\4", s)

        # Pass 5: lowercase and collapse all whitespace
        return " ".join(s.lower().split())
