"""TreeNode dataclass and NodeType StrEnum for JSON-to-tree representation.

Provides the foundational data types used by TreeBuilder (Plan 02) to convert
JSON documents into typed tree structures for semantic comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Any


class NodeType(StrEnum):
    """Enumeration of the five structural node types in a JSON tree.

    StrEnum values are the lowercased member names (Python 3.11+):
    - OBJECT   -> "object"  : JSON object {}
    - KEY      -> "key"     : A key within a JSON object
    - ARRAY    -> "array"   : JSON array []
    - ELEMENT  -> "element" : An element within a JSON array
    - SCALAR   -> "scalar"  : A leaf value (string, number, bool, null)
    """

    OBJECT = auto()
    KEY = auto()
    ARRAY = auto()
    ELEMENT = auto()
    SCALAR = auto()


@dataclass(slots=True)
class TreeNode:
    """A node in the JSON tree representation.

    Attributes:
        node_type:  Which kind of node this is (see NodeType).
        label:      Normalized key for KEY nodes; str(value) for SCALAR nodes;
                    empty string for structural nodes (OBJECT, ARRAY, ELEMENT).
        path:       JSON Pointer path (RFC 6901), e.g. "/user/name".
        raw_label:  Original (un-normalized) key for KEY nodes; empty for all others.
        value:      Original typed Python value for SCALAR nodes; None for structural.
        children:   Child nodes. Must use field(default_factory=list) to ensure
                    each instance gets its own independent list.
    """

    node_type: NodeType
    label: str
    path: str
    raw_label: str = ""
    value: Any = None
    children: list[TreeNode] = field(default_factory=list)


def subtree_size(node: TreeNode) -> int:
    """Return the total number of nodes in the subtree rooted at ``node``.

    Counts the node itself plus every transitive child.  Used by the STED
    algorithm to scale edit costs for whole-subtree delete/insert operations
    (e.g. when two compared nodes have different ``node_type``) so the
    resulting cost reflects how much structure was destroyed and created,
    not a flat 1.0.

    Args:
        node: The root of the subtree to measure.

    Returns:
        Integer count >= 1 (the node itself always counts).
    """
    total = 1
    # Iterative DFS to avoid recursion limits on very deep trees
    stack: list[TreeNode] = list(node.children)
    while stack:
        current = stack.pop()
        total += 1
        if current.children:
            stack.extend(current.children)
    return total
