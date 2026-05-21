"""STEDConfig and ArrayComparisonMode for STED algorithm configuration.

STEDConfig is a frozen (immutable) dataclass holding the algorithm
parameters.  ArrayComparisonMode selects how arrays are compared:
ordered (positional), unordered (set-like), or auto-detected.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto


class ArrayComparisonMode(StrEnum):
    """How to compare JSON arrays during STED computation.

    - ORDERED:   Positional alignment via DP sequence edit distance.
    - UNORDERED: Set-like matching via Hungarian algorithm.
    - AUTO:      Infer from array content (scalars → unordered, objects → ordered).
    """

    ORDERED = auto()
    UNORDERED = auto()
    AUTO = auto()


@dataclass(frozen=True, slots=True)
class STEDConfig:
    """Immutable configuration for the STED algorithm.

    Attributes:
        w_s: Structural weight in [0, 1].
        w_c: Content weight in [0, 1].  Must satisfy w_s + w_c ≈ 1.0.
        lambda_unmatched: Penalty multiplier for unmatched children (≥ 0).
        array_comparison_mode: How arrays are compared.
        type_coercion: When True, numeric strings are coerced to numbers before
            content comparison (e.g. "123" == 123 -> distance 0.0).  Default False.
        null_equals_missing: When True, a JSON null value is treated as equivalent
            to a missing key.  Default False.
        ignore_paths: Tuple of JSON-Pointer-style path patterns that select
            sub-trees to drop from BOTH inputs before comparison.  Useful for
            excluding volatile keys such as timestamps, generated ids, or
            version numbers.  Default ``()`` (no paths ignored).

            Path syntax (v1):

            * Patterns are rooted (must start with ``/``).
            * Components are slash-separated, matching the JSON Pointer paths
              that :class:`json_semantic_diff.tree.builder.TreeBuilder` emits
              (e.g. ``/users/0/id``).
            * A single ``*`` component matches exactly one path component
              (object key or array index).  For example ``/users/*/id``
              matches ``/users/0/id``, ``/users/1/id``, etc.
            * The pattern must point at an OBJECT KEY (not at an array
              element).  On a match, the key is removed from its parent
              object before tree building.  Targeting array indices directly
              is not supported in v1 — use ``*`` to skip an index level.
            * Patterns may not be empty, may not end in ``/``, and may not
              contain empty path components (e.g. ``/a//b``).
    """

    w_s: float = 0.5
    w_c: float = 0.5
    lambda_unmatched: float = 0.1
    array_comparison_mode: ArrayComparisonMode = ArrayComparisonMode.ORDERED
    type_coercion: bool = False
    null_equals_missing: bool = False
    ignore_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.w_s <= 1.0:
            msg = f"w_s must be in [0, 1], got {self.w_s}"
            raise ValueError(msg)
        if not 0.0 <= self.w_c <= 1.0:
            msg = f"w_c must be in [0, 1], got {self.w_c}"
            raise ValueError(msg)
        if abs(self.w_s + self.w_c - 1.0) >= 1e-9:
            msg = f"w_s + w_c must sum to 1.0, got {self.w_s + self.w_c}"
            raise ValueError(msg)
        if self.lambda_unmatched < 0.0:
            msg = f"lambda_unmatched must be >= 0.0, got {self.lambda_unmatched}"
            raise ValueError(msg)
        # ignore_paths is typed as a tuple, but users sometimes pass a list at
        # runtime — reject explicitly so the failure mode is a clear TypeError
        # rather than a confusing mid-comparison crash.
        ignore_paths_obj: object = self.ignore_paths
        if not isinstance(ignore_paths_obj, tuple):
            msg = (
                f"ignore_paths must be a tuple of strings, "
                f"got {type(ignore_paths_obj).__name__}"
            )
            raise TypeError(msg)
        for pattern in self.ignore_paths:
            _validate_ignore_path(pattern)


def _validate_ignore_path(pattern: object) -> None:
    """Validate a single ignore_paths pattern.

    Patterns must:

    * be ``str`` instances,
    * start with ``/``,
    * be longer than a single ``/`` (the root is not a valid target),
    * not end in ``/``,
    * contain no empty path components (``/a//b`` is rejected).

    Raises:
        TypeError: If the pattern is not a string.
        ValueError: If the pattern violates any of the syntax rules.
    """
    if not isinstance(pattern, str):
        msg = (
            f"ignore_paths entries must be strings, "
            f"got {type(pattern).__name__}: {pattern!r}"
        )
        raise TypeError(msg)
    if not pattern.startswith("/"):
        msg = f"ignore_paths entry must start with '/', got {pattern!r}"
        raise ValueError(msg)
    if pattern == "/":
        msg = (
            "ignore_paths entry '/' is not a valid target — "
            "patterns must select a key, not the root"
        )
        raise ValueError(msg)
    if pattern.endswith("/"):
        msg = f"ignore_paths entry must not end with '/', got {pattern!r}"
        raise ValueError(msg)
    # pattern.split("/") with leading "/" yields a leading "" component
    # we expect: ["", "a", "b", ...]; any other "" means an empty component
    components = pattern.split("/")[1:]
    if any(c == "" for c in components):
        msg = f"ignore_paths entry has empty path component, got {pattern!r}"
        raise ValueError(msg)
