"""Public API functions for json-semantic-diff.

This module provides the user-facing functions: compare, compare_batch,
compare_batch_pairs, consistency_score, is_equivalent, and similarity_score.
The single-call entry points create a fresh STEDComparator (or
ConsistencyScorer) to guarantee zero global state mutation between calls.
The batch entry points create ONE comparator and reuse it across every pair
so the embedding cache amortises across the batch.
"""

from __future__ import annotations

from typing import Any

from json_semantic_diff.algorithm.config import STEDConfig
from json_semantic_diff.comparator import STEDComparator
from json_semantic_diff.result import ComparisonResult
from json_semantic_diff.scorer import ConsistencyScorer

__all__ = [
    "compare",
    "compare_batch",
    "compare_batch_pairs",
    "consistency_score",
    "format_diff",
    "is_equivalent",
    "similarity_score",
]


def consistency_score(
    docs: list[Any],
    config: STEDConfig | None = None,
) -> float:
    """Return a consistency score measuring how stable a generator is across samples.

    The score penalizes both low average similarity (generator produces different
    outputs) and high variance (generator is erratic). Formula:
    ``max(0, mean(pairwise_scores) - std(pairwise_scores))``.

    Args:
        docs:   List of JSON values to score for consistency. A generator that
                always produces the same output will score 1.0; an erratic
                generator with high variance will score close to 0.0.
        config: Algorithm hyper-parameters. Defaults to ``STEDConfig()`` when None.

    Returns:
        A float in [0.0, 1.0]. 1.0 means all documents are identical (perfectly
        consistent). Returns 1.0 for empty lists and single-document lists.

    Raises:
        ValueError: If ``config`` is provided with invalid weights (e.g.
            ``w_s + w_c != 1.0`` or ``lambda_unmatched < 0``).  Raised by
            :class:`STEDConfig` on construction.
    """
    return ConsistencyScorer(config=config).compute(docs)


def compare(
    left: Any,
    right: Any,
    config: STEDConfig | None = None,
) -> ComparisonResult:
    """Compare two JSON values and return a rich ComparisonResult.

    Creates a fresh ``STEDComparator`` per call to guarantee zero global state
    mutation between calls.

    To exclude volatile keys (timestamps, generated ids, version numbers) from
    the comparison, pass ``STEDConfig(ignore_paths=("/timestamp", "/users/*/id"))``.
    See :class:`json_semantic_diff.algorithm.config.STEDConfig` for the full
    pattern syntax.

    Args:
        left:   First JSON value (dict, list, str, int, float, bool, None).
        right:  Second JSON value.
        config: Algorithm hyper-parameters. Defaults to ``STEDConfig()`` when None.

    Returns:
        A ``ComparisonResult`` with similarity_score, matched_pairs, key_mappings,
        unmatched_left, unmatched_right, and computation_time_ms populated.
        Paths in ``ignore_paths`` are stripped before tree construction, so they
        will not appear in any of these fields.

    Raises:
        TypeError: If ``left`` or ``right`` is not a JSON value (dict, list,
            str, int, float, bool, None).
        ValueError: If ``config`` is provided with invalid weights (e.g.
            ``w_s + w_c != 1.0`` or ``lambda_unmatched < 0``) or invalid
            ``ignore_paths`` patterns.  Raised by :class:`STEDConfig` on
            construction.
    """
    comparator = STEDComparator(config=config)
    return comparator.compare(left, right)


def is_equivalent(
    left: Any,
    right: Any,
    threshold: float = 0.85,
    config: STEDConfig | None = None,
) -> bool:
    """Return True if the two JSON values are semantically equivalent.

    Two values are considered equivalent when their similarity score is at or
    above the given threshold. The default threshold of 0.85 is tuned to pass
    benign naming differences (camelCase vs snake_case) while rejecting genuine
    structural breaks.

    Args:
        left:      First JSON value.
        right:     Second JSON value.
        threshold: Minimum similarity score to consider equivalent. Must be in
                   [0.0, 1.0]. Defaults to 0.85.
        config:    Algorithm hyper-parameters. Defaults to ``STEDConfig()`` when None.

    Returns:
        True if ``compare(left, right, config).similarity_score >= threshold``.

    Raises:
        TypeError: If ``left`` or ``right`` is not a JSON value.
        ValueError: If ``threshold`` is outside ``[0.0, 1.0]``, or if ``config``
            is provided with invalid weights.
    """
    if not 0.0 <= threshold <= 1.0:
        msg = f"threshold must be in [0.0, 1.0], got {threshold}"
        raise ValueError(msg)
    result = compare(left, right, config=config)
    return result.similarity_score >= threshold


def compare_batch(
    lefts: list[Any],
    right: Any,
    config: STEDConfig | None = None,
) -> list[ComparisonResult]:
    """Compare each value in ``lefts`` against a single ``right`` value.

    A single ``STEDComparator`` is constructed and reused across every pair,
    so the embedding cache amortises across the batch — for ``N`` lefts that
    share KEY labels with ``right``, the backend's ``embed()`` is typically
    invoked only on the first pair.

    Args:
        lefts:  List of JSON values to compare against ``right``.  May be
                empty (in which case an empty list is returned).
        right:  The shared right-hand JSON value.
        config: Algorithm hyper-parameters.  Defaults to ``STEDConfig()`` when
                None.

    Returns:
        A list of ``ComparisonResult`` objects, one per element of ``lefts``,
        in the same order as the input.

    Raises:
        TypeError: If any element of ``lefts``, or ``right``, is not a JSON
            value (dict, list, str, int, float, bool, None).
        ValueError: If ``config`` is provided with invalid weights.
    """
    if not lefts:
        return []
    comparator = STEDComparator(config=config)
    return [comparator.compare(left, right) for left in lefts]


def compare_batch_pairs(
    pairs: list[tuple[Any, Any]],
    config: STEDConfig | None = None,
) -> list[ComparisonResult]:
    """Compare a list of (left, right) pairs, returning results in input order.

    A single ``STEDComparator`` is constructed and reused across every pair,
    so the embedding cache amortises across the batch.

    Args:
        pairs:  List of ``(left, right)`` JSON value tuples.  May be empty
                (in which case an empty list is returned).
        config: Algorithm hyper-parameters.  Defaults to ``STEDConfig()`` when
                None.

    Returns:
        A list of ``ComparisonResult`` objects, one per input pair, in the
        same order as ``pairs``.

    Raises:
        TypeError: If any element of any pair is not a JSON value.
        ValueError: If ``config`` is provided with invalid weights.
    """
    if not pairs:
        return []
    comparator = STEDComparator(config=config)
    return [comparator.compare(left, right) for left, right in pairs]


def format_diff(result: ComparisonResult, indent: int = 2) -> str:
    """Render a :class:`ComparisonResult` as a human-friendly text report.

    Produces a deterministic side-by-side summary with three optional
    sections: matched key pairs, keys unmatched on the left, and keys
    unmatched on the right.  Empty sections are skipped entirely (their
    headers are not printed) so the output stays tight on identical or
    near-identical inputs.

    Output shape::

        Similarity: 0.87  (computed in 1.3 ms)

        Matched (3):
          user_name <-> userName
          email_address <-> emailAddress
          age <-> age

        Unmatched in left (2):
          /timestamp
          /metadata/version

        Unmatched in right (1):
          /createdAt

    Determinism: matched pairs are sorted alphabetically by the LEFT
    label (the last path component of the left key); unmatched paths are
    sorted alphabetically.  This means two runs over the same inputs
    always produce byte-identical output (modulo the timing line).

    Args:
        result: A ``ComparisonResult`` from
            :func:`compare`.
        indent: Number of leading spaces on each item line.  Defaults to
            ``2``.  Negative values are clamped to ``0``.

    Returns:
        A multi-line string ending without a trailing newline.

    Raises:
        TypeError: If ``indent`` is not an ``int``.
    """
    if not isinstance(indent, int) or isinstance(indent, bool):
        msg = f"indent must be an int, got {type(indent).__name__}"
        raise TypeError(msg)
    pad = " " * max(indent, 0)

    lines: list[str] = [
        f"Similarity: {result.similarity_score:.2f}  "
        f"(computed in {result.computation_time_ms:.1f} ms)"
    ]

    # Matched section: sort by the trailing path segment of the left key
    # so the section order is independent of insertion order.
    if result.matched_pairs:
        sorted_pairs = sorted(
            result.matched_pairs,
            key=lambda pair: (_last_segment(pair[0]), _last_segment(pair[1])),
        )
        lines.append("")
        lines.append(f"Matched ({len(sorted_pairs)}):")
        for left_path, right_path in sorted_pairs:
            lines.append(
                f"{pad}{_last_segment(left_path)} <-> {_last_segment(right_path)}"
            )

    if result.unmatched_left:
        lines.append("")
        lines.append(f"Unmatched in left ({len(result.unmatched_left)}):")
        for path in sorted(result.unmatched_left):
            lines.append(f"{pad}{path}")

    if result.unmatched_right:
        lines.append("")
        lines.append(f"Unmatched in right ({len(result.unmatched_right)}):")
        for path in sorted(result.unmatched_right):
            lines.append(f"{pad}{path}")

    return "\n".join(lines)


def _last_segment(path: str) -> str:
    """Return the trailing path segment of a JSON Pointer-style path.

    ``"/users/0/email"`` -> ``"email"``.  Empty / root paths return as-is
    so the caller can still tell something went wrong.
    """
    if not path or path == "/":
        return path
    return path.rsplit("/", 1)[-1]


def similarity_score(
    left: Any,
    right: Any,
    config: STEDConfig | None = None,
) -> float:
    """Return the normalised similarity score for two JSON values.

    Args:
        left:   First JSON value.
        right:  Second JSON value.
        config: Algorithm hyper-parameters. Defaults to ``STEDConfig()`` when None.

    Returns:
        A float in [0.0, 1.0]. 1.0 means identical; 0.0 means completely dissimilar.

    Raises:
        TypeError: If ``left`` or ``right`` is not a JSON value.
        ValueError: If ``config`` is provided with invalid weights.
    """
    result = compare(left, right, config=config)
    return result.similarity_score
