"""Cost functions for STED tree edit distance.

Implements the STED paper cost formula:
    gamma_upd = w_s * gamma_struct + w_c * gamma_content

- cost_insert / cost_delete: unit cost (1.0) for inserting/deleting a node.
- cost_update: blended structural + content distance for substitution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from json_semantic_diff.algorithm.config import STEDConfig
from json_semantic_diff.tree.nodes import NodeType, TreeNode

if TYPE_CHECKING:
    from json_semantic_diff.protocols import EmbeddingBackend


def cost_insert(node: TreeNode) -> float:
    """Unit cost for inserting a node."""
    return 1.0


def cost_delete(node: TreeNode) -> float:
    """Unit cost for deleting a node."""
    return 1.0


def _structural_similarity(
    node_a: TreeNode,
    node_b: TreeNode,
    backend: EmbeddingBackend,
) -> float:
    """Compute structural similarity between two nodes.

    For KEY nodes: uses backend.similarity if available, else embed + cosine.
    For non-KEY nodes: 1.0 if same NodeType, 0.0 if different.
    """
    # Both must be KEY for label-based comparison
    if node_a.node_type == NodeType.KEY and node_b.node_type == NodeType.KEY:
        if hasattr(backend, "similarity"):
            return float(backend.similarity(node_a.label, node_b.label))
        # Fallback: embed + cosine
        embeddings = backend.embed([node_a.label, node_b.label])
        vec_a = embeddings[0]
        vec_b = embeddings[1]
        dot = float(np.dot(vec_a, vec_b))
        norm_a = float(np.linalg.norm(vec_a))
        norm_b = float(np.linalg.norm(vec_b))
        return dot / (norm_a * norm_b + 1e-9)

    # Non-KEY nodes: type match = structurally identical
    if node_a.node_type == node_b.node_type:
        return 1.0
    return 0.0


def _content_distance(node_a: TreeNode, node_b: TreeNode, config: STEDConfig) -> float:
    """Compute content distance between two nodes.

    SCALAR nodes: 0.0 if values equal, 1.0 otherwise.

    POLICY: ``bool`` is its own type — never coerced.  ``True == 1`` and
    ``False == 0`` are TRUE in Python (bool subclasses int), which would
    spuriously collapse ``True`` and ``1`` to distance 0.0.  We guard
    against that with a ``type(a) is type(b)`` check before equality, and
    explicitly disqualify bools from the numeric-coercion fast-path.

    When ``config.type_coercion`` is True, numeric strings are coerced
    before comparison (e.g. ``"123" == 123`` yields 0.0) — but again,
    only when neither side is a bool.

    Non-SCALAR nodes: 0.0 (structural nodes have no content to compare).
    """
    if node_a.node_type == NodeType.SCALAR and node_b.node_type == NodeType.SCALAR:
        va = node_a.value
        vb = node_b.value

        # Bool is its own type — exact-type identity required for equality.
        # This prevents True == 1 / False == 0 from collapsing to 0.0.
        a_is_bool = isinstance(va, bool)
        b_is_bool = isinstance(vb, bool)
        if a_is_bool or b_is_bool:
            if a_is_bool and b_is_bool and va == vb:
                return 0.0
            return 1.0

        # Non-bool: ordinary equality (same-type values, e.g. "5" == "5",
        # 5 == 5, 5.0 == 5).  Different types like "5" vs 5 are NOT equal
        # unless the type-coercion path below accepts them.
        if type(va) is type(vb) and va == vb:
            return 0.0

        # Same numeric value across int/float (e.g. 5 == 5.0) — int/float
        # are mutually compatible numeric types, no coercion needed.
        if type(va) in (int, float) and type(vb) in (int, float) and va == vb:
            return 0.0

        if config.type_coercion:
            try:
                # Coerce only when at least one side is numeric (int/float)
                # AND neither side is a bool.  Already excluded by the
                # a_is_bool/b_is_bool branch above.
                if type(va) in (int, float) or type(vb) in (int, float):
                    return 0.0 if float(va) == float(vb) else 1.0
            except (ValueError, TypeError):
                pass
        return 1.0
    return 0.0


def cost_update(
    node_a: TreeNode,
    node_b: TreeNode,
    backend: EmbeddingBackend,
    config: STEDConfig,
) -> float:
    """Compute update cost between two nodes using w_s/w_c blending.

    Returns:
        Float in [0, 1]: ``config.w_s * gamma_struct + config.w_c * gamma_content``
    """
    gamma_struct = 1.0 - _structural_similarity(node_a, node_b, backend)
    gamma_content = _content_distance(node_a, node_b, config)
    return config.w_s * gamma_struct + config.w_c * gamma_content
