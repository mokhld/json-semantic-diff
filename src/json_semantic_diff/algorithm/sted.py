"""STEDAlgorithm: recursive tree edit distance for JSON semantic comparison.

Implements the STED (Semantic Tree Edit Distance) algorithm that traverses
two JSON trees simultaneously and computes a similarity score in [0, 1].

Architecture:
- OBJECT nodes:  Children (KEY nodes) matched via Hungarian algorithm (order-invariant).
- ARRAY nodes:   Children (ELEMENT nodes) matched via ordered DP, unordered Hungarian,
                 or auto-detected based on child homogeneity.
- KEY nodes:     key-label cost (cost_update) + recursive value child distance.
- ELEMENT nodes: Transparent wrapper — distance = value child distance.
- SCALAR nodes:  cost_update for exact/type-aware value comparison.

Per-level normalization (STED paper formula) is applied after each child
matching step so that deep nesting does not bias the overall score.

The critical invariant: ``_compute_node_distance`` returns a raw distance
(not normalized, may be > 1 for structural nodes).  Normalization is applied
by the *caller* at the appropriate child-list level.  The public ``compute``
method is the only place a normalized [0, 1] similarity is returned directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from json_semantic_diff.algorithm.config import ArrayComparisonMode, STEDConfig
from json_semantic_diff.algorithm.costs import cost_delete, cost_insert, cost_update
from json_semantic_diff.algorithm.matcher import hungarian_match
from json_semantic_diff.algorithm.normalizer import normalize_similarity
from json_semantic_diff.result import NodeContribution
from json_semantic_diff.tree.builder import TreeBuilder
from json_semantic_diff.tree.nodes import NodeType, TreeNode, subtree_size

if TYPE_CHECKING:
    from json_semantic_diff.protocols import EmbeddingBackend


class STEDAlgorithm:
    """Recursive STED algorithm for JSON semantic similarity.

    Accepts any two valid JSON values and returns a similarity score in
    [0.0, 1.0] — 1.0 means structurally and semantically identical,
    0.0 means completely unrelated.

    Example::

        from json_semantic_diff.algorithm.sted import STEDAlgorithm
        from json_semantic_diff.backends import StaticBackend

        algo = STEDAlgorithm(backend=StaticBackend())
        score = algo.compute({"user_name": "Alice"}, {"userName": "Alice"})
        # score > 0.85  (naming-convention equivalents)
    """

    def __init__(
        self,
        backend: EmbeddingBackend,
        config: STEDConfig | None = None,
    ) -> None:
        """Initialise the algorithm with an embedding backend and configuration.

        Args:
            backend: An EmbeddingBackend-conformant object used for key-label
                similarity computation.  The algorithm never imports concrete
                backend classes — only the Protocol is used at type-check time.
            config:  Algorithm hyper-parameters.  Defaults to ``STEDConfig()``
                (w_s=0.5, w_c=0.5, lambda_unmatched=0.1, mode=ORDERED).
        """
        self._backend = backend
        self._config = config if config is not None else STEDConfig()
        self._builder = TreeBuilder()
        # Memoisation cache for `_compute_node_distance`.  Keyed on
        # ``(id(node_a), id(node_b), depth)``: TreeNode identity is stable for
        # the duration of a single ``compute()`` call (trees are built once,
        # then walked read-only), and ``depth`` matters because the
        # ``max_depth`` cap short-circuits results past the threshold.
        # Cleared at the top of every ``compute()`` call so cached values from
        # previous comparisons never leak across calls — see audit finding I1.
        self._dist_cache: dict[tuple[int, int, int], float] = {}
        # Explain-mode buffer.  When ``config.collect_explanation`` is False
        # (default) this stays ``None`` and every "is the buffer live?" check
        # is a single ``is None`` test — no per-call list allocation, no
        # branch on the hot path beyond a None comparison.  When True,
        # ``compute()`` swaps it to a fresh list at the top and the
        # recursion appends ``NodeContribution`` entries at meaningful
        # boundaries.  After ``compute()`` finishes the list is sorted by
        # contribution descending and exposed via ``last_explanations``.
        self._explanations: list[NodeContribution] | None = None
        # Stable handle for callers (comparator) to read after compute().
        # Starts empty; replaced on each compute() when explain mode is on.
        self.last_explanations: tuple[NodeContribution, ...] = ()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, json_a: Any, json_b: Any) -> float:
        """Compute semantic similarity between two JSON values.

        Args:
            json_a: First JSON value (dict, list, str, int, float, bool, None).
            json_b: Second JSON value (dict, list, str, int, float, bool, None).

        Returns:
            Float in [0.0, 1.0] — similarity score.
        """
        # Defensive: clear the distance cache so prior comparisons cannot
        # contaminate this one.  ``id()`` is reused by CPython for freed
        # objects, so leaking cache entries across ``compute()`` calls would
        # be a correctness bug rather than a perf nuisance.
        self._dist_cache.clear()
        # Reset the explain buffer for this run.  When the config flag is
        # off, the buffer stays ``None`` and the recursion's "buffer alive?"
        # checks short-circuit immediately — bit-identical behaviour to the
        # pre-explain implementation.
        if self._config.collect_explanation:
            self._explanations = []
        else:
            self._explanations = None
        self.last_explanations = ()

        root_a = self._builder.build(json_a)
        root_b = self._builder.build(json_b)
        score = self._compute_similarity(root_a, root_b)

        # Explain mode: cover the two cases that never reach a child-
        # matching loop and would otherwise leave the buffer empty:
        # - scalar-vs-scalar roots that differ, and
        # - type-mismatched roots (e.g. {} vs []) which are flagged
        #   maximally dissimilar at the very top of _compute_similarity.
        if self._explanations is not None and not self._explanations:
            self._record_root_contribution(root_a, root_b)

        # Sort + freeze the collected explanations for the caller.  Highest
        # contribution first so users scanning the list see the most
        # score-moving differences at the top.
        if self._explanations is not None:
            self._explanations.sort(key=lambda c: c.contribution, reverse=True)
            self.last_explanations = tuple(self._explanations)
        return score

    def _record_root_contribution(self, root_a: TreeNode, root_b: TreeNode) -> None:
        """Emit a single root-level contribution when no child matching ran.

        Two situations leave the explain buffer empty even though the
        score < 1.0:

        * Type-mismatched roots — ``_compute_similarity`` returns 0.0
          straight away.
        * Scalar-vs-scalar roots whose values differ — the cost is
          captured in the SCALAR cost_update path, which child matching
          never invokes.

        Both are reported as a single ``value_mismatch`` at ``"/"`` so
        users see the top-level explanation instead of an empty list.
        Identical roots (score 1.0) are skipped — the recursion already
        confirmed there's nothing to say.
        """
        assert self._explanations is not None
        if root_a.node_type != root_b.node_type:
            # Use the larger subtree size as the contribution magnitude so
            # type-flipped roots show up with weight proportional to the
            # work they replaced.
            cost = float(max(subtree_size(root_a), subtree_size(root_b)))
            if cost > 0.0:
                self._explanations.append(
                    NodeContribution(
                        path="/",
                        contribution=cost,
                        kind="value_mismatch",
                        detail=f"type {root_a.node_type} vs {root_b.node_type}",
                    )
                )
            return
        if root_a.node_type == NodeType.SCALAR:
            cost = cost_update(root_a, root_b, self._backend, self._config)
            if cost > 0.0:
                self._explanations.append(
                    NodeContribution(
                        path="/",
                        contribution=cost,
                        kind="value_mismatch",
                        detail="",
                    )
                )

    # ------------------------------------------------------------------
    # Core similarity dispatcher (normalizes at structural boundaries)
    # ------------------------------------------------------------------

    def _compute_similarity(self, node_a: TreeNode, node_b: TreeNode) -> float:
        """Compute similarity between two tree nodes, returning [0, 1].

        This is the normalized view of the distance computation.  Structural
        nodes (OBJECT, ARRAY) normalize the raw child-matching distance.
        Leaf nodes (SCALAR) convert cost_update distance to similarity directly.
        KEY and ELEMENT nodes use their own normalization logic.

        Args:
            node_a: Left tree node.
            node_b: Right tree node.

        Returns:
            Float in [0.0, 1.0] similarity score.
        """
        # Type mismatch: maximally dissimilar
        if node_a.node_type != node_b.node_type:
            return 0.0

        node_type = node_a.node_type

        if node_type == NodeType.SCALAR:
            # SCALAR: direct cost → similarity conversion
            dist = cost_update(node_a, node_b, self._backend, self._config)
            return 1.0 - dist

        if node_type == NodeType.KEY:
            # KEY: per-level normalization over 1 child (the value)
            return self._compute_key_similarity(node_a, node_b)

        if node_type == NodeType.ELEMENT:
            # ELEMENT: transparent — similarity = similarity of value child
            return self._compute_element_similarity(node_a, node_b)

        if node_type == NodeType.OBJECT:
            # OBJECT: normalize raw Hungarian cost over KEY children
            return self._compute_object_similarity(node_a, node_b)

        # ARRAY: normalize raw ordered/unordered cost over ELEMENT children
        # (This is the final NodeType variant — NodeType has exactly 5 members)
        return self._compute_array_similarity(node_a, node_b)

    # ------------------------------------------------------------------
    # Raw distance computation (unnormalized, used for cost matrices)
    # ------------------------------------------------------------------

    def _compute_node_distance(
        self,
        node_a: TreeNode,
        node_b: TreeNode,
        depth: int = 0,
    ) -> float:
        """Compute raw edit distance between two nodes.

        Returns a non-negative distance value (may exceed 1.0 for structural
        nodes with multiple children).  Callers are responsible for
        normalizing via ``normalize_similarity``.

        - SCALAR: cost_update in [0, 1].
        - KEY: key-label cost + value-child distance (sum, range [0, 2]).
        - ELEMENT: value-child distance (range [0, max_child_depth]).
        - OBJECT: raw Hungarian cost over KEY children (range [0, n_children]).
        - ARRAY: raw DP/Hungarian cost over ELEMENT children.
        - Type mismatch: returns max cost based on the larger node.

        Max-depth policy: when ``config.max_depth`` is set and ``depth >=
        max_depth``, no further recursion happens.  Instead the sub-tree
        contributes ``cost_delete + cost_insert`` (= 2.0) — i.e. the two
        sub-trees are treated as fully unrelated past the cap.  This
        bounds traversal cost at the price of resolution past ``max_depth``
        levels below the roots.

        Args:
            node_a: Left tree node.
            node_b: Right tree node.
            depth: Current recursion depth from the comparison roots
                (root call uses ``depth=0``).  Incremented when stepping
                into a child's children.

        Returns:
            Non-negative float distance.
        """
        # Memoisation: Hungarian cost-matrix construction (and ordered DP) call
        # this for every (m * n) child pair, often visiting the same
        # (node_a, node_b) repeatedly when keys point at structurally identical
        # subtrees.  Memoising on ``(id(node_a), id(node_b), depth)`` collapses
        # the O(n⁴) worst case (audit finding I1) without changing semantics.
        # ``depth`` is part of the key because the ``max_depth`` cap (checked
        # below) makes the result depth-dependent; including it is always
        # correct, even when ``max_depth is None``.
        # Identity shortcut: same object compared with itself is always 0.0.
        # (Falls through to the cache so callers still benefit from the
        # uniform lookup path on repeat queries.)
        cache_key = (id(node_a), id(node_b), depth)
        cached = self._dist_cache.get(cache_key)
        if cached is not None:
            return cached

        result = self._compute_node_distance_uncached(node_a, node_b, depth)
        self._dist_cache[cache_key] = result
        return result

    def _compute_node_distance_uncached(
        self,
        node_a: TreeNode,
        node_b: TreeNode,
        depth: int,
    ) -> float:
        """Uncached body of ``_compute_node_distance``.

        Split out so the public entry point can do a single cache lookup at
        the top and a single store at the bottom, instead of threading the
        cache through every early-return branch.  Recursive calls go back
        through ``_compute_node_distance`` (the cached entry point) so deeper
        sub-trees still hit the cache.
        """
        # Max-depth cap: short-circuit before any structural work so deep
        # sub-trees do not pay traversal cost.  We charge the standard
        # "unrelated pair" cost (delete + insert) — equivalent to declining
        # to compare further.  Identical-shape trivial leaves at the cap
        # still go through their normal cost path below because the cap is
        # checked *before* recursing into children, not at every node.
        max_depth = self._config.max_depth
        if max_depth is not None and depth >= max_depth:
            return cost_delete(node_a) + cost_insert(node_b)

        # Type mismatch: full delete of the left subtree + full insert of the
        # right subtree.  Cost scales with subtree size so a deeply-different
        # branch correctly outweighs a single-scalar swap during Hungarian /
        # DP matching.  Using ``max`` (rather than sum) keeps the per-node
        # contribution proportional to the larger structure being replaced.
        if node_a.node_type != node_b.node_type:
            return float(max(subtree_size(node_a), subtree_size(node_b)))

        node_type = node_a.node_type

        if node_type == NodeType.SCALAR:
            return cost_update(node_a, node_b, self._backend, self._config)

        if node_type == NodeType.KEY:
            # Key-label edit cost
            key_label_dist = cost_update(node_a, node_b, self._backend, self._config)
            # Recursive value-child distance
            if node_a.children and node_b.children:
                val_dist = self._compute_node_distance(
                    node_a.children[0], node_b.children[0], depth + 1
                )
            else:
                # Malformed KEY (no child): same = 0, different = 1
                val_dist = 0.0 if (not node_a.children and not node_b.children) else 1.0
            return key_label_dist + val_dist

        if node_type == NodeType.ELEMENT:
            # ELEMENT is transparent: distance = its single value-child distance
            if node_a.children and node_b.children:
                return self._compute_node_distance(
                    node_a.children[0], node_b.children[0], depth + 1
                )
            return 0.0 if (not node_a.children and not node_b.children) else 1.0

        if node_type == NodeType.OBJECT:
            if not node_a.children and not node_b.children:
                return 0.0
            return self._match_children_hungarian(
                node_a.children, node_b.children, depth + 1
            )

        # ARRAY (final variant)
        if not node_a.children and not node_b.children:
            return 0.0
        mode = self._resolve_array_mode(node_a, node_b)
        if mode == ArrayComparisonMode.ORDERED:
            return self._match_children_sequence(
                node_a.children, node_b.children, depth + 1
            )
        return self._match_children_hungarian(
            node_a.children, node_b.children, depth + 1
        )

    # ------------------------------------------------------------------
    # Normalized similarity helpers for each structural node type
    # ------------------------------------------------------------------

    def _compute_key_similarity(self, key_a: TreeNode, key_b: TreeNode) -> float:
        """Compute normalized similarity between two KEY nodes.

        A KEY node wraps exactly one value child, but that value can be a
        deep subtree (object/array).  The raw distance returned by
        ``_compute_node_distance`` sums the key-label cost ([0, 1]) and the
        recursive value distance (which scales with subtree size — easily
        > 1 for nested structures).  Normalising by ``len(children)`` (≤ 1)
        would clip almost every non-trivial difference to 0.0, collapsing
        KEY similarity to binary — see audit finding C6.

        Instead, normalise by the maximum *subtree size* of the two value
        children plus 1 (for the key-label cost itself).  That keeps the
        ratio in [0, 1] without losing resolution: a small leaf change in a
        large subtree no longer wipes out the entire score.

        Args:
            key_a: Left KEY node.
            key_b: Right KEY node.

        Returns:
            Float in [0.0, 1.0].
        """
        raw_dist = self._compute_node_distance(key_a, key_b)
        # Denominator = 1 (the key-label cost) + max size of the two value
        # subtrees.  Empty / malformed KEY nodes fall back to size 0.
        size_a = subtree_size(key_a.children[0]) if key_a.children else 0
        size_b = subtree_size(key_b.children[0]) if key_b.children else 0
        denom = 1 + max(size_a, size_b)
        return normalize_similarity(
            raw_dist,
            n_left=denom,
            n_right=denom,
            lambda_=self._config.lambda_unmatched,
        )

    def _compute_element_similarity(self, elem_a: TreeNode, elem_b: TreeNode) -> float:
        """Compute similarity between two ELEMENT nodes.

        ELEMENT nodes are transparent wrappers around array values.
        Delegates entirely to the single child comparison.

        Args:
            elem_a: Left ELEMENT node.
            elem_b: Right ELEMENT node.

        Returns:
            Float in [0.0, 1.0] similarity score.
        """
        if elem_a.children and elem_b.children:
            return self._compute_similarity(elem_a.children[0], elem_b.children[0])
        # Both empty — identical; one empty, one not — no match.
        return 1.0 if (not elem_a.children and not elem_b.children) else 0.0

    def _compute_object_similarity(self, obj_a: TreeNode, obj_b: TreeNode) -> float:
        """Compute similarity between two OBJECT nodes via Hungarian matching.

        Matches KEY children optimally (order-invariant), normalizes the
        resulting raw distance via the STED paper formula (child-count
        denominator preserves backwards-compatible scoring for shallow
        disjoint objects).

        Args:
            obj_a: Left OBJECT node.
            obj_b: Right OBJECT node.

        Returns:
            Float in [0.0, 1.0] similarity score.
        """
        children_a = obj_a.children
        children_b = obj_b.children

        # Both empty objects: identical
        if not children_a and not children_b:
            return 1.0

        raw_dist = self._match_children_hungarian(children_a, children_b)
        return normalize_similarity(
            raw_dist,
            len(children_a),
            len(children_b),
            self._config.lambda_unmatched,
        )

    def _compute_array_similarity(self, arr_a: TreeNode, arr_b: TreeNode) -> float:
        """Compute similarity between two ARRAY nodes.

        Dispatches to ordered (DP) or unordered (Hungarian) matching based
        on the resolved array comparison mode.

        Args:
            arr_a: Left ARRAY node.
            arr_b: Right ARRAY node.

        Returns:
            Float in [0.0, 1.0] similarity score.
        """
        children_a = arr_a.children
        children_b = arr_b.children

        # Both empty arrays: identical
        if not children_a and not children_b:
            return 1.0

        mode = self._resolve_array_mode(arr_a, arr_b)

        if mode == ArrayComparisonMode.ORDERED:
            raw_dist = self._match_children_sequence(children_a, children_b)
        else:
            raw_dist = self._match_children_hungarian(children_a, children_b)

        return normalize_similarity(
            raw_dist,
            len(children_a),
            len(children_b),
            self._config.lambda_unmatched,
        )

    # ------------------------------------------------------------------
    # Explain-mode helpers
    # ------------------------------------------------------------------

    def _record_pair_contribution(
        self,
        node_a: TreeNode,
        node_b: TreeNode,
        cost: float,
    ) -> None:
        """Append a single matched-pair contribution to the explain buffer.

        Classifies the pair:

        * Both SCALAR (or one side wraps a SCALAR via KEY/ELEMENT) with
          non-zero cost → ``value_mismatch`` (the values differ at this
          location).
        * Anything else with non-zero cost → ``matched`` (a matched
          structural pair that still carries some distance because their
          sub-trees differ).

        The buffer must already be live (caller's responsibility).  ``cost``
        is the raw per-pair distance, identical to the one summed into the
        final score.
        """
        assert self._explanations is not None  # invariant from caller
        # Resolve the "leaf" each side reduces to so we can detect a pure
        # value mismatch.  KEY/ELEMENT nodes are transparent — their value
        # child carries the actual scalar (or sub-tree).
        leaf_a = self._scalar_leaf(node_a)
        leaf_b = self._scalar_leaf(node_b)
        if (
            leaf_a is not None
            and leaf_b is not None
            and leaf_a.node_type == NodeType.SCALAR
            and leaf_b.node_type == NodeType.SCALAR
        ):
            kind = "value_mismatch"
        else:
            kind = "matched"
        # Use the left-side path when available; falls back to the right.
        # Both should be non-empty for any non-root node.
        path = node_a.path or node_b.path or "/"
        self._explanations.append(
            NodeContribution(
                path=path,
                contribution=cost,
                kind=kind,
                detail="",
            )
        )

    @staticmethod
    def _scalar_leaf(node: TreeNode) -> TreeNode | None:
        """Return the SCALAR descendant of a transparent wrapper, or None.

        KEY and ELEMENT nodes wrap exactly one child.  Step through them
        once: if the wrapped child is a SCALAR, return it; otherwise
        return None so callers fall back to the structural-match branch.
        """
        if node.node_type == NodeType.SCALAR:
            return node
        if node.node_type in (NodeType.KEY, NodeType.ELEMENT) and node.children:
            child = node.children[0]
            if child.node_type == NodeType.SCALAR:
                return child
        return None

    # ------------------------------------------------------------------
    # Child matching strategies
    # ------------------------------------------------------------------

    def _match_children_hungarian(
        self,
        children_a: list[TreeNode],
        children_b: list[TreeNode],
        depth: int = 0,
    ) -> float:
        """Optimal bipartite child matching via Hungarian algorithm.

        Builds a distance cost matrix (m x n) where cell [i][j] is the
        distance between children_a[i] and children_b[j], then finds the
        minimum-cost assignment.  Unmatched children contribute unit costs.

        Args:
            children_a: Children of the left node.
            children_b: Children of the right node.
            depth: Recursion depth at which the children live (one deeper
                than their parent).  Threaded through so ``max_depth``
                short-circuits cost-matrix entries past the cap.

        Returns:
            Total raw matching distance (not normalized).
        """
        m = len(children_a)
        n = len(children_b)

        if m == 0 and n == 0:
            return 0.0

        # Build cost matrix from raw node distances
        cost_matrix = np.empty((m, n), dtype=float)
        for i, ca in enumerate(children_a):
            for j, cb in enumerate(children_b):
                cost_matrix[i, j] = self._compute_node_distance(ca, cb, depth)

        row_ind, col_ind = hungarian_match(cost_matrix)

        matched_cost = (
            float(cost_matrix[row_ind, col_ind].sum()) if len(row_ind) else 0.0
        )

        # Unmatched children contribute unit insert/delete costs
        matched_left = set(row_ind.tolist())
        matched_right = set(col_ind.tolist())

        unmatched_left_cost = sum(
            cost_delete(children_a[i]) for i in range(m) if i not in matched_left
        )
        unmatched_right_cost = sum(
            cost_insert(children_b[j]) for j in range(n) if j not in matched_right
        )

        # Explain mode: record contributions for the winning matches and
        # for the unmatched leftovers.  Cheap when off (single ``is None``
        # check) and never re-runs the cost matrix — we read directly off
        # the assignment computed above.
        if self._explanations is not None:
            for r, c in zip(row_ind.tolist(), col_ind.tolist(), strict=True):
                pair_cost = float(cost_matrix[r, c])
                if pair_cost > 0.0:
                    self._record_pair_contribution(
                        children_a[r], children_b[c], pair_cost
                    )
            for i in range(m):
                if i not in matched_left:
                    self._explanations.append(
                        NodeContribution(
                            path=children_a[i].path or "/",
                            contribution=cost_delete(children_a[i]),
                            kind="unmatched_left",
                            detail=children_a[i].raw_label or children_a[i].label,
                        )
                    )
            for j in range(n):
                if j not in matched_right:
                    self._explanations.append(
                        NodeContribution(
                            path=children_b[j].path or "/",
                            contribution=cost_insert(children_b[j]),
                            kind="unmatched_right",
                            detail=children_b[j].raw_label or children_b[j].label,
                        )
                    )

        return matched_cost + unmatched_left_cost + unmatched_right_cost

    def _match_children_sequence(
        self,
        children_a: list[TreeNode],
        children_b: list[TreeNode],
        depth: int = 0,
    ) -> float:
        """Ordered sequence alignment via DP edit distance.

        Computes the minimum-cost alignment of two ordered child sequences.
        Insert cost (add from right) and delete cost (remove from left) are
        each 1.0.  Substitution cost is the raw node-pair distance.

        Args:
            children_a: Children of the left node (ordered).
            children_b: Children of the right node (ordered).
            depth: Recursion depth at which the children live (one deeper
                than their parent).  Threaded through so ``max_depth``
                short-circuits substitution costs past the cap.

        Returns:
            Total raw alignment distance (not normalized).
        """
        m = len(children_a)
        n = len(children_b)

        # dp[i][j] = min cost to align children_a[:i] with children_b[:j]
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]
        # Substitution-cost mirror used by the explain-mode backtrack so we
        # don't have to call ``_compute_node_distance`` a second time
        # (it's cached, but skipping the call entirely keeps backtracking
        # branch-free).  ``None`` when explain mode is off.
        sub_costs: list[list[float]] | None = (
            [[0.0] * (n + 1) for _ in range(m + 1)]
            if self._explanations is not None
            else None
        )

        # Base cases: aligning with empty sequence
        for i in range(1, m + 1):
            dp[i][0] = dp[i - 1][0] + cost_delete(children_a[i - 1])
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j - 1] + cost_insert(children_b[j - 1])

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                sub_cost = self._compute_node_distance(
                    children_a[i - 1], children_b[j - 1], depth
                )
                if sub_costs is not None:
                    sub_costs[i][j] = sub_cost
                dp[i][j] = min(
                    dp[i - 1][j] + cost_delete(children_a[i - 1]),  # delete
                    dp[i][j - 1] + cost_insert(children_b[j - 1]),  # insert
                    dp[i - 1][j - 1] + sub_cost,  # substitute
                )

        # Explain mode: backtrack the optimal alignment and record per-
        # position contributions (substitution, deletion, insertion).
        # Cheap when off — guarded by an ``is None`` check.
        if self._explanations is not None and sub_costs is not None:
            self._backtrack_sequence(children_a, children_b, dp, sub_costs)

        return dp[m][n]

    def _backtrack_sequence(
        self,
        children_a: list[TreeNode],
        children_b: list[TreeNode],
        dp: list[list[float]],
        sub_costs: list[list[float]],
    ) -> None:
        """Walk the DP table back along the chosen alignment edges.

        Emits one :class:`NodeContribution` per non-zero edit on the
        winning path.  Called only when explain mode is on; the buffer is
        guaranteed live by the caller, so we don't re-check
        ``self._explanations is not None`` here.

        Tie-breaking matches the forward pass's ``min`` order: prefer the
        substitute branch when costs tie, then delete, then insert.  This
        keeps the recorded alignment deterministic across runs.
        """
        assert self._explanations is not None  # invariant from caller
        i = len(children_a)
        j = len(children_b)
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                sub = dp[i - 1][j - 1] + sub_costs[i][j]
                if dp[i][j] == sub:
                    cost = sub_costs[i][j]
                    if cost > 0.0:
                        self._record_pair_contribution(
                            children_a[i - 1], children_b[j - 1], cost
                        )
                    i -= 1
                    j -= 1
                    continue
            if i > 0:
                delete = dp[i - 1][j] + cost_delete(children_a[i - 1])
                if dp[i][j] == delete:
                    self._explanations.append(
                        NodeContribution(
                            path=children_a[i - 1].path or "/",
                            contribution=cost_delete(children_a[i - 1]),
                            kind="unmatched_left",
                            detail=children_a[i - 1].raw_label
                            or children_a[i - 1].label,
                        )
                    )
                    i -= 1
                    continue
            if j > 0:
                self._explanations.append(
                    NodeContribution(
                        path=children_b[j - 1].path or "/",
                        contribution=cost_insert(children_b[j - 1]),
                        kind="unmatched_right",
                        detail=children_b[j - 1].raw_label or children_b[j - 1].label,
                    )
                )
                j -= 1

    # ------------------------------------------------------------------
    # Array mode resolution
    # ------------------------------------------------------------------

    def _resolve_array_mode(
        self, arr_a: TreeNode, arr_b: TreeNode
    ) -> ArrayComparisonMode:
        """Determine the effective array comparison mode.

        If the config specifies ORDERED or UNORDERED, that value is returned
        directly.  If AUTO, the array contents are inspected:
        - All ELEMENT children contain only SCALAR values -> UNORDERED.
        - Any ELEMENT child contains an OBJECT or ARRAY value -> ORDERED.
        - Empty arrays -> UNORDERED (both empty = identical regardless of mode).

        Args:
            arr_a: Left ARRAY node.
            arr_b: Right ARRAY node.

        Returns:
            Resolved ArrayComparisonMode (never AUTO).
        """
        mode = self._config.array_comparison_mode
        if mode != ArrayComparisonMode.AUTO:
            return mode

        # AUTO: inspect all elements from both arrays
        all_elements = arr_a.children + arr_b.children

        if not all_elements:
            return ArrayComparisonMode.UNORDERED

        for elem in all_elements:
            if not elem.children:
                continue
            child_type = elem.children[0].node_type
            if child_type in (NodeType.OBJECT, NodeType.ARRAY):
                return ArrayComparisonMode.ORDERED

        # All elements contain scalars (or are empty)
        return ArrayComparisonMode.UNORDERED
