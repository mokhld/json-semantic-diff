"""HungarianMatcher: optimal bipartite assignment with np.inf guard.

Wraps scipy's ``linear_sum_assignment`` so that infinite-cost cells never
reach the solver (which would raise ``ValueError``).  After assignment,
pairs that landed on originally-infinite positions are filtered out.

Guard value formula: ``finite_max * 2.0 + 1.0``

This module also exposes :func:`build_alias_set` and
:func:`aliased_key_similarity`, the alias-aware KEY similarity hook used
by :class:`json_semantic_diff.comparator.STEDComparator` to short-circuit
user-declared key-equivalence pairs (e.g. ``("uid", "user_id")``) to a
similarity of ``1.0`` before the embedding backend is consulted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

from json_semantic_diff.tree.normalizer import KeyNormalizer

if TYPE_CHECKING:
    from json_semantic_diff.protocols import EmbeddingBackend

# Module-level singleton — KeyNormalizer is stateless and cheap to share.
# Used to expand user-supplied alias pairs into a backend-normalised form
# so that, e.g., ``aliases=(("id", "user_id"),)`` matches ``user-id``
# (which normalises to the same canonical form as ``user_id``).
_alias_normalizer = KeyNormalizer()


def build_alias_set(
    aliases: tuple[tuple[str, str], ...],
) -> frozenset[frozenset[str]]:
    """Build an O(1) lookup set of canonical alias pairs.

    Each user-supplied ``(a, b)`` pair contributes up to TWO frozensets:
    one for the raw labels (so we match the user's spelling exactly) and
    one for the labels after :class:`KeyNormalizer` normalisation (so
    aliases survive camelCase/snake_case/kebab-case rewrites that the
    default backend applies before scoring).

    Pairs where ``a == b`` (or where both sides normalise to the same
    string) contribute nothing useful — they would self-match with or
    without the alias hint.  We still include them for explicit-intent
    documentation; the lookup is O(1) either way.

    Args:
        aliases: Validated tuple of ``(str, str)`` pairs from
            :class:`STEDConfig`.

    Returns:
        A ``frozenset`` of 2-element ``frozenset[str]`` entries.  Empty
        when ``aliases`` is empty.
    """
    if not aliases:
        return frozenset()
    pairs: set[frozenset[str]] = set()
    for a, b in aliases:
        pairs.add(frozenset({a, b}))
        norm_a = _alias_normalizer.normalize(a)
        norm_b = _alias_normalizer.normalize(b)
        pairs.add(frozenset({norm_a, norm_b}))
    return frozenset(pairs)


def aliased_key_similarity(
    backend: EmbeddingBackend,
    label_a: str,
    label_b: str,
    alias_set: frozenset[frozenset[str]],
) -> float:
    """Return KEY similarity with alias short-circuit.

    If ``(label_a, label_b)`` (or any equivalent pair after
    :class:`KeyNormalizer` normalisation) is present in ``alias_set``,
    short-circuits to ``1.0`` without consulting the backend.  Otherwise
    delegates to ``backend.similarity(label_a, label_b)``.

    Args:
        backend: Embedding backend exposing ``similarity(a, b) -> float``.
        label_a: First key label.  May be raw or backend-normalised; the
            check covers both.
        label_b: Second key label.
        alias_set: Lookup set built via :func:`build_alias_set`.

    Returns:
        Float in ``[0.0, 1.0]`` — ``1.0`` for aliased pairs, otherwise
        the backend's own similarity verdict.
    """
    if alias_set:
        if frozenset({label_a, label_b}) in alias_set:
            return 1.0
        # Defensive: handle the case where the matcher receives one raw
        # label and one normalised label, or vice versa.  Normalising
        # both sides costs ~one regex sweep per call and keeps the
        # short-circuit predictable across backends.
        norm_a = _alias_normalizer.normalize(label_a)
        norm_b = _alias_normalizer.normalize(label_b)
        if frozenset({norm_a, norm_b}) in alias_set:
            return 1.0
    return float(backend.similarity(label_a, label_b))


def hungarian_match(
    cost_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute optimal bipartite assignment with np.inf guard.

    Args:
        cost_matrix: 2-D cost matrix of shape ``(m, n)``.  May contain
            ``np.inf`` to mark forbidden assignments.

    Returns:
        Tuple ``(row_ind, col_ind)`` of 1-D integer arrays giving the
        optimal assignment, with any pair whose *original* cost was
        infinite removed.  Empty arrays are returned when no valid
        assignment exists.
    """
    if cost_matrix.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    cost = np.asarray(cost_matrix, dtype=float)

    inf_mask = np.isinf(cost)

    # All-inf: no valid assignment
    if inf_mask.all():
        return np.array([], dtype=int), np.array([], dtype=int)

    # Replace inf with a guard value that dominates all finite costs
    if inf_mask.any():
        finite_max = float(cost[~inf_mask].max())
        guard_value = finite_max * 2.0 + 1.0
        cost = np.where(inf_mask, guard_value, cost)

    row_ind, col_ind = linear_sum_assignment(cost)

    # Filter out pairs whose original cost was infinite
    if inf_mask.any():
        original = np.asarray(cost_matrix, dtype=float)
        keep = np.isfinite(original[row_ind, col_ind])
        row_ind = row_ind[keep]
        col_ind = col_ind[keep]

    return row_ind, col_ind
