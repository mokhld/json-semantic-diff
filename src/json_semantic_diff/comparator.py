"""STEDComparator: orchestrator that wires TreeBuilder + STEDAlgorithm + EmbeddingBackend.

This is the central wiring layer between the raw algorithm and the
public API.  It converts a raw float score into a rich ComparisonResult
with matched pairs, key mappings, timing data, and null_equals_missing preprocessing.

Architecture:
- compare() starts a wall-clock timer, preprocesses inputs, builds trees for
  match extraction, delegates scoring to STEDAlgorithm.compute(), extracts
  KEY-level match data via Hungarian matching, and returns a ComparisonResult.
- Trees are built TWICE: once internally by STEDAlgorithm.compute() (for scoring)
  and once here (for match extraction).  This is intentional — the comparator
  must not mutate the algorithm's internal state.
- null_equals_missing=True is implemented as a preprocessing step that strips
  None-valued keys from dicts before both the score computation and tree building.
- Embedding results are cached via EmbeddingCache (LRU).  All unique KEY labels
  from both trees are pre-warmed into the cache in a single embed() call before
  the algorithm runs — guaranteeing one embed call per unique label set.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from json_semantic_diff.algorithm.config import STEDConfig
from json_semantic_diff.algorithm.matcher import (
    aliased_key_similarity,
    build_alias_set,
    hungarian_match,
)
from json_semantic_diff.algorithm.sted import STEDAlgorithm
from json_semantic_diff.backends import StaticBackend
from json_semantic_diff.cache import EmbeddingCache
from json_semantic_diff.result import ComparisonResult
from json_semantic_diff.tree.builder import TreeBuilder
from json_semantic_diff.tree.nodes import NodeType, TreeNode

if TYPE_CHECKING:
    from json_semantic_diff.protocols import EmbeddingBackend

__all__ = ["STEDComparator"]


def _path_matches(pattern: str, path: str) -> bool:
    """Return True if ``path`` matches ``pattern``.

    Both ``pattern`` and ``path`` are JSON-Pointer-style strings ("/a/b/c").
    A single ``*`` component in ``pattern`` matches exactly one path component
    in ``path`` (no partial-component glob matching, no multi-component
    wildcards).  Component counts must match exactly — this is a key-level
    selector, not a prefix match.

    Args:
        pattern: Configured ignore-path pattern (already validated by
            :func:`STEDConfig._validate_ignore_path`).
        path:    Candidate JSON Pointer path for a dict key.

    Returns:
        True if every component lines up, False otherwise.
    """
    pat_parts = pattern.split("/")[1:]
    path_parts = path.split("/")[1:]
    if len(pat_parts) != len(path_parts):
        return False
    for p, c in zip(pat_parts, path_parts, strict=True):
        if p == "*":
            continue
        if p != c:
            return False
    return True


class STEDComparator:
    """Orchestrator for semantic JSON comparison.

    Wires ``TreeBuilder``, ``STEDAlgorithm``, and an ``EmbeddingBackend``
    together into a single ``compare()`` call that returns a rich
    ``ComparisonResult`` with similarity score, matched pairs, key mappings,
    unmatched paths, and wall-clock timing.

    Embedding results are automatically cached using an LRU cache.  The first
    ``compare()`` call pre-warms the cache with all unique KEY labels from both
    trees in a single ``embed()`` call.  Subsequent ``compare()`` calls on the
    same documents produce **zero** backend embed calls.

    Two separate ``STEDComparator`` instances never share cache state — each
    instance maintains its own ``EmbeddingCache``.

    Example::

        from json_semantic_diff.comparator import STEDComparator

        cmp = STEDComparator()
        result = cmp.compare({"user_name": "Alice"}, {"userName": "Alice"})
        print(result.similarity_score)   # > 0.85
        print(result.key_mappings)       # {"user_name": "userName"}
    """

    def __init__(
        self,
        backend: EmbeddingBackend | None = None,
        config: STEDConfig | None = None,
        max_cache_size: int = 512,
    ) -> None:
        """Initialise the comparator.

        Args:
            backend: An EmbeddingBackend-conformant object.  Defaults to
                ``StaticBackend()`` when None.
            config:  Algorithm hyper-parameters.  Defaults to ``STEDConfig()``.
            max_cache_size: Maximum number of string embeddings held in the
                per-instance LRU cache.  When exceeded, the least-recently-used
                entry is silently evicted.  Defaults to 512.
                This is an infrastructure parameter — it is NOT part of
                ``STEDConfig`` (which governs algorithm behaviour only).
        """
        self._config: STEDConfig = config if config is not None else STEDConfig()
        raw_backend: Any = backend if backend is not None else StaticBackend()
        self._backend: EmbeddingCache = EmbeddingCache(
            raw_backend, max_size=max_cache_size
        )
        self._algorithm = STEDAlgorithm(backend=self._backend, config=self._config)
        self._builder = TreeBuilder()
        # Pre-build the alias lookup once per comparator so the hot loop in
        # _walk_object_pair pays O(1) per pair instead of re-normalising on
        # every cell of the cost matrix.
        self._alias_set: frozenset[frozenset[str]] = build_alias_set(
            self._config.aliases
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare(self, left: Any, right: Any) -> ComparisonResult:
        """Compare two JSON values and return a rich ComparisonResult.

        The comparison is stateless with respect to the *algorithm*: calling
        this method twice with the same inputs and config will always produce
        identical result values.  (The cache is a performance detail that does
        not affect correctness.)

        Args:
            left:  First JSON value (dict, list, str, int, float, bool, None).
            right: Second JSON value.

        Returns:
            A ``ComparisonResult`` with all six fields populated.

        Raises:
            TypeError: If either ``left`` or ``right`` is not a JSON value
                (dict, list, str, int, float, bool, None).  Only the top-level
                type is validated here — nested invalid types will surface
                deeper in the algorithm.
        """
        self._validate_json_type(left, "left")
        self._validate_json_type(right, "right")

        t0 = time.perf_counter()

        # Preprocess: always return fresh structures, never mutate input
        left = self._preprocess(left)
        right = self._preprocess(right)

        # Build trees ONCE for match extraction (STEDAlgorithm builds its own)
        left_tree = self._builder.build(left)
        right_tree = self._builder.build(right)

        # Batch pre-warm the cache with all unique KEY labels.
        # Collects every KEY node label from both trees and embeds them in a
        # single backend call.  Subsequent algorithm.compute() and
        # _walk_object_pair() calls will serve all embeddings/similarities
        # from the warm cache — zero additional backend embed() calls.
        all_labels = self._collect_key_labels(left_tree) | self._collect_key_labels(
            right_tree
        )
        if all_labels:
            self._backend.embed(
                list(all_labels)
            )  # populates cache; return value unused

        # Score from algorithm (builds its own trees internally)
        score = self._algorithm.compute(left, right)

        # Extract KEY-level match data from our trees
        matched_pairs, key_mappings, unmatched_left, unmatched_right = (
            self._extract_key_matches(left_tree, right_tree)
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return ComparisonResult(
            similarity_score=score,
            matched_pairs=tuple(matched_pairs),
            key_mappings=key_mappings,
            unmatched_left=tuple(unmatched_left),
            unmatched_right=tuple(unmatched_right),
            computation_time_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_json_type(value: Any, name: str) -> None:
        """Validate that ``value`` is a top-level JSON type.

        Rejects datetimes, sets, custom class instances, and anything else
        that ``json.dumps`` would reject.  Nested invalid types are NOT
        checked here — they surface lazily during tree construction.

        Args:
            value: The value to validate.
            name:  Argument name (used in the error message).

        Raises:
            TypeError: If ``value`` is not a JSON value.
        """
        # bool is a subclass of int, but both are valid JSON — no special-case needed.
        if value is None or isinstance(value, (dict, list, str, int, float, bool)):
            return
        msg = (
            f"compare() expects JSON values (dict, list, str, int, float, bool, None) "
            f"for {name}; got {type(value).__name__}"
        )
        raise TypeError(msg)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, value: Any) -> Any:
        """Return a fresh copy of ``value``, applying configured preprocessing.

        Always returns a freshly constructed structure for ``dict`` and ``list``
        inputs — the caller never sees a reference to the original container.
        This guarantees that comparison is non-mutating regardless of which
        preprocessing options are enabled.

        Preprocessing applied (in order):

        1. ``null_equals_missing``: when True, recursively removes any dict
           entry whose value is None so ``{"x": None}`` becomes ``{}``.
        2. ``ignore_paths``: any object key whose path matches one of the
           configured patterns is dropped from its parent object.  Patterns
           are matched against the JSON Pointer path the key would have in
           the resulting tree (e.g. ``/users/0/id``); single-component ``*``
           wildcards in patterns match exactly one path component.  Array
           elements themselves are never removed — patterns are applied at
           the key level only.

        Args:
            value: Any valid JSON value.

        Returns:
            A fresh structure mirroring ``value`` (immutable scalars are
            returned as-is — they cannot be mutated).
        """
        return self._preprocess_inner(value, path="")

    def _preprocess_inner(self, value: Any, path: str) -> Any:
        """Recursive worker for ``_preprocess`` that threads the current path."""
        strip_nones = self._config.null_equals_missing
        ignore_paths = self._config.ignore_paths

        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for k, v in value.items():
                if strip_nones and v is None:
                    continue
                # Only consider string keys for path matching; non-string
                # keys (which TreeBuilder later rejects) are passed through
                # unchanged so the rejection still surfaces with the original
                # error message.
                if isinstance(k, str):
                    child_path = f"{path}/{k}"
                    if any(_path_matches(p, child_path) for p in ignore_paths):
                        continue
                    out[k] = self._preprocess_inner(v, child_path)
                else:
                    out[k] = self._preprocess_inner(v, path)
            return out
        if isinstance(value, list):
            return [
                self._preprocess_inner(item, f"{path}/{idx}")
                for idx, item in enumerate(value)
            ]
        return value

    # ------------------------------------------------------------------
    # Cache pre-warming helpers
    # ------------------------------------------------------------------

    def _collect_key_labels(self, node: TreeNode) -> set[str]:
        """Recursively collect all KEY node labels from a tree.

        Uses ``node.label`` (normalized form) rather than ``node.raw_label``
        because the algorithm's cost functions pass ``node.label`` to
        ``backend.embed()`` / ``backend.similarity()``.  Pre-scanning with
        the same strings ensures cache hits on every subsequent lookup.

        Args:
            node: Root (or any sub-root) of a JSON tree.

        Returns:
            Set of normalized label strings for every KEY node in the subtree.
        """
        labels: set[str] = set()
        if node.node_type == NodeType.KEY:
            labels.add(node.label)  # MUST use .label (normalized), NOT .raw_label
        for child in node.children:
            labels |= self._collect_key_labels(child)
        return labels

    # ------------------------------------------------------------------
    # Match extraction
    # ------------------------------------------------------------------

    def _extract_key_matches(
        self,
        left_tree: TreeNode,
        right_tree: TreeNode,
    ) -> tuple[list[tuple[str, str]], dict[str, str], list[str], list[str]]:
        """Extract KEY-level match data via Hungarian matching.

        Recursively walks both trees simultaneously.  At each OBJECT node
        pair, builds a cost matrix over KEY children using backend similarity,
        runs Hungarian assignment, and collects matched/unmatched KEY paths.

        Args:
            left_tree:  Root of the left JSON tree.
            right_tree: Root of the right JSON tree.

        Returns:
            A 4-tuple ``(matched_pairs, key_mappings, unmatched_left, unmatched_right)``:
            - matched_pairs: List of ``(left_key_path, right_key_path)`` tuples.
            - key_mappings:  Dict mapping raw left key name → raw right key name.
            - unmatched_left:  JSON Pointer paths for unmatched left KEY nodes.
            - unmatched_right: JSON Pointer paths for unmatched right KEY nodes.
        """
        matched_pairs: list[tuple[str, str]] = []
        key_mappings: dict[str, str] = {}
        unmatched_left: list[str] = []
        unmatched_right: list[str] = []

        self._walk_object_pair(
            left_tree,
            right_tree,
            matched_pairs,
            key_mappings,
            unmatched_left,
            unmatched_right,
        )

        return matched_pairs, key_mappings, unmatched_left, unmatched_right

    def _walk_object_pair(
        self,
        left: TreeNode,
        right: TreeNode,
        matched_pairs: list[tuple[str, str]],
        key_mappings: dict[str, str],
        unmatched_left: list[str],
        unmatched_right: list[str],
    ) -> None:
        """Recursively walk a pair of nodes, matching KEY children at OBJECT nodes.

        Args:
            left, right: Nodes at the same structural position in each tree.
            matched_pairs, key_mappings, unmatched_left, unmatched_right:
                Accumulator collections (mutated in place).
        """
        # Only process OBJECT/OBJECT pairs for KEY matching
        if left.node_type != NodeType.OBJECT or right.node_type != NodeType.OBJECT:
            return

        left_keys = left.children  # KEY nodes
        right_keys = right.children  # KEY nodes

        # Edge case: both empty
        if not left_keys and not right_keys:
            return

        # Edge case: one side empty
        if not left_keys:
            unmatched_right.extend(k.path for k in right_keys)
            return
        if not right_keys:
            unmatched_left.extend(k.path for k in left_keys)
            return

        # Build cost matrix: distance(left_key_i, right_key_j) = 1 - similarity
        m = len(left_keys)
        n = len(right_keys)
        cost_matrix = np.empty((m, n), dtype=float)

        for i, ka in enumerate(left_keys):
            for j, kb in enumerate(right_keys):
                # _backend is always an EmbeddingCache (constructed in __init__),
                # which guarantees similarity() is available — either delegated
                # to the wrapped backend or computed via cosine of embed().
                # aliased_key_similarity short-circuits to 1.0 when (ka, kb)
                # match a user-declared alias pair; otherwise it delegates
                # straight through to the cached backend.
                key_sim = aliased_key_similarity(
                    self._backend, ka.label, kb.label, self._alias_set
                )
                cost_matrix[i, j] = 1.0 - key_sim

        row_ind, col_ind = hungarian_match(cost_matrix)

        matched_left_indices = set(row_ind.tolist())
        matched_right_indices = set(col_ind.tolist())

        # Record matched pairs
        for r, c in zip(row_ind.tolist(), col_ind.tolist(), strict=True):
            lk = left_keys[r]
            rk = right_keys[c]
            matched_pairs.append((lk.path, rk.path))
            key_mappings[lk.raw_label] = rk.raw_label

            # Recurse into value children if both are OBJECT nodes
            if lk.children and rk.children:
                lk_val = lk.children[0]
                rk_val = rk.children[0]
                if (
                    lk_val.node_type == NodeType.OBJECT
                    and rk_val.node_type == NodeType.OBJECT
                ):
                    self._walk_object_pair(
                        lk_val,
                        rk_val,
                        matched_pairs,
                        key_mappings,
                        unmatched_left,
                        unmatched_right,
                    )

        # Unmatched left keys
        for i in range(m):
            if i not in matched_left_indices:
                unmatched_left.append(left_keys[i].path)

        # Unmatched right keys
        for j in range(n):
            if j not in matched_right_indices:
                unmatched_right.append(right_keys[j].path)
