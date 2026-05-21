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
        numeric_tolerance: Absolute tolerance for numeric scalar comparison.
            When both compared values are numeric (``int``/``float`` but
            NOT ``bool``), they are treated as equal (content distance
            ``0.0``) if ``abs(a - b) <= numeric_tolerance``.  Must be
            ``>= 0.0``.  Default ``0.0`` preserves exact-equality
            semantics so existing scores are unchanged.

            Interaction with ``type_coercion``: when one side is a numeric
            string and ``type_coercion=True``, the string is coerced first
            and the tolerance is then applied to the resulting floats.
        max_depth: Optional cap on tree-traversal recursion depth.  When
            set, sub-trees deeper than ``max_depth`` levels below the roots
            are not compared recursively — their contents contribute a
            shallow ``cost_delete + cost_insert`` (treated as unrelated
            unless trivially identical at the cap).

            Trade-off: a smaller ``max_depth`` yields faster comparisons
            (especially on deep nested structures) at the cost of losing
            resolution past that depth — two large sub-trees with a tiny
            deep-leaf difference will score identically to two completely
            disjoint sub-trees once the cap is hit.

            Must be ``None`` (no cap, full traversal) or ``>= 1`` (a depth
            of ``0`` would short-circuit at the roots and is rejected).
            Default ``None`` preserves the existing full-recursion
            behaviour.
        aliases: Tuple of ``(canonical, alias)`` string pairs that the
            KEY-matching layer must treat as equivalent (similarity
            ``1.0``).  The relation is symmetric — ``(("uid",
            "user_id"),)`` makes ``uid`` match ``user_id`` AND
            ``user_id`` match ``uid``.  Each entry must be a 2-tuple of
            non-empty strings; otherwise a :class:`ValueError` (or
            :class:`TypeError` for non-tuple containers) is raised.
            Default ``()`` (no aliases).

            Aliases short-circuit the backend's similarity verdict for
            KEY comparison only — they do not affect content-level
            value comparison.  The check is performed both on raw
            user-supplied labels and on the backend-normalised forms,
            so an alias pair ``("id", "user_id")`` correctly matches
            ``{"id": 1}`` against ``{"user-id": 1}`` (since the
            backend normalises ``user-id`` to the same canonical form
            as ``user_id``).
    """

    w_s: float = 0.5
    w_c: float = 0.5
    lambda_unmatched: float = 0.1
    array_comparison_mode: ArrayComparisonMode = ArrayComparisonMode.ORDERED
    type_coercion: bool = False
    null_equals_missing: bool = False
    ignore_paths: tuple[str, ...] = ()
    numeric_tolerance: float = 0.0
    max_depth: int | None = None
    aliases: tuple[tuple[str, str], ...] = ()

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
        # numeric_tolerance: must be a non-negative float.  bool is a
        # subclass of int — reject it explicitly so config(numeric_tolerance=True)
        # doesn't silently mean tolerance=1.0.
        tol_obj: object = self.numeric_tolerance
        if isinstance(tol_obj, bool) or not isinstance(tol_obj, (int, float)):
            msg = (
                f"numeric_tolerance must be a non-negative number, "
                f"got {type(tol_obj).__name__}: {tol_obj!r}"
            )
            raise TypeError(msg)
        if self.numeric_tolerance < 0.0:
            msg = f"numeric_tolerance must be >= 0.0, got {self.numeric_tolerance}"
            raise ValueError(msg)
        # max_depth: None means uncapped; otherwise must be a positive int.
        # bool is a subclass of int — reject it for the same reason as above.
        max_depth_obj: object = self.max_depth
        if max_depth_obj is not None:
            if isinstance(max_depth_obj, bool) or not isinstance(max_depth_obj, int):
                msg = (
                    f"max_depth must be None or a positive int, "
                    f"got {type(max_depth_obj).__name__}: {max_depth_obj!r}"
                )
                raise TypeError(msg)
            if self.max_depth is not None and self.max_depth < 1:
                msg = f"max_depth must be None or >= 1, got {self.max_depth}"
                raise ValueError(msg)
        # aliases: must be a tuple of 2-tuples of non-empty strings.
        aliases_obj: object = self.aliases
        if not isinstance(aliases_obj, tuple):
            msg = (
                f"aliases must be a tuple of (str, str) pairs, "
                f"got {type(aliases_obj).__name__}"
            )
            raise TypeError(msg)
        for entry in self.aliases:
            _validate_alias_pair(entry)


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


def _validate_alias_pair(entry: object) -> None:
    """Validate a single aliases entry.

    Each entry must be a 2-tuple of non-empty strings.  Anything else —
    a non-tuple, wrong arity, non-string elements, or empty strings —
    raises :class:`ValueError`.

    Args:
        entry: The candidate alias pair.

    Raises:
        ValueError: If the entry is not a 2-tuple of non-empty strings.
    """
    if not isinstance(entry, tuple):
        msg = (
            f"aliases entries must be 2-tuples of strings, "
            f"got {type(entry).__name__}: {entry!r}"
        )
        raise ValueError(msg)
    if len(entry) != 2:
        msg = f"aliases entry must have exactly 2 elements, got {len(entry)}: {entry!r}"
        raise ValueError(msg)
    a, b = entry
    if not isinstance(a, str) or not isinstance(b, str):
        msg = f"aliases entry elements must be strings, got {entry!r}"
        raise ValueError(msg)
    if not a or not b:
        msg = f"aliases entry elements must be non-empty strings, got {entry!r}"
        raise ValueError(msg)
