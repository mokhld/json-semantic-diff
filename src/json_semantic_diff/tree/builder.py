"""TreeBuilder: converts any valid JSON value into a typed TreeNode tree.

Uses recursive dispatch to convert JSON dicts, lists, and scalar values into
a tree of TreeNode objects. Object keys are normalized via KeyNormalizer.
Array elements are wrapped in ELEMENT nodes with index-based labels.
Scalar values preserve their original Python type in the TreeNode.value field.

JSON Pointer paths (RFC 6901) are built during traversal:
- Root is "" (empty string)
- Each level appends "/{escaped_key_or_index}"
- Per RFC 6901 section 4, key segments are escaped: ``~`` becomes ``~0``
  and ``/`` becomes ``~1``.  The ``~`` substitution is applied first so
  that a literal ``~`` in a key does not eat the ``~`` produced by the
  ``/`` substitution.  Array indices are decimal integers and need no
  escaping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from json_semantic_diff.tree.nodes import NodeType, TreeNode
from json_semantic_diff.tree.normalizer import KeyNormalizer

# Module-level normalizer (stateless, safe to share across all TreeBuilder instances)
_normalizer = KeyNormalizer()

# Type alias for valid JSON values
JsonValue = dict[str, Any] | list[Any] | str | int | float | bool | None


def _escape_path_segment(segment: str) -> str:
    """Escape a single JSON Pointer reference token per RFC 6901 section 4.

    The order matters: replace ``~`` first (``~`` → ``~0``), then ``/``
    (``/`` → ``~1``).  Reversing the order would cause a literal ``~`` in
    the input to corrupt the ``~1`` produced by an earlier ``/`` escape.

    Args:
        segment: A raw object key (already known to be a ``str``).

    Returns:
        The escaped reference token, safe to concatenate after ``"/"``.
    """
    return segment.replace("~", "~0").replace("/", "~1")


@dataclass
class TreeBuilder:
    """Converts any valid JSON value into a typed TreeNode tree.

    Uses recursive dispatch to handle all JSON types. The dispatch order is
    critical: bool MUST be checked before int because bool is a subclass of int
    in Python (isinstance(True, int) is True).

    JSON Pointer paths (RFC 6901):
        Root node has path="" (empty string).
        Each nested level appends "/{escaped_key_or_index}" to the parent
        path.  Object keys are escaped per RFC 6901 section 4 (``~`` → ``~0``,
        ``/`` → ``~1``, ``~`` substitution first).  Array indices are decimal
        integers and require no escaping.

    Type preservation:
        SCALAR nodes store the original Python value in the `value` field.
        "5" (str) and 5 (int) produce distinct nodes with the same label "5"
        but different type(node.value). The STED algorithm uses this to detect
        type mismatches between compared documents.

    Example::
        builder = TreeBuilder()
        tree = builder.build({"user_name": "John"})
        # tree: OBJECT -> KEY("user name", raw="user_name") -> SCALAR("John")
    """

    def build(self, value: JsonValue, path: str = "") -> TreeNode:
        """Convert a JSON value to a TreeNode tree.

        Cycle detection: a ``visited`` set of ``id(container)`` values
        tracks every dict / list seen on the current call stack so that a
        self-referencing Python object (e.g. ``a = {}; a["self"] = a``)
        raises ``ValueError`` instead of recursing forever.  Scalars do
        not participate — only containers (dict/list) can form a cycle.

        Args:
            value: Any valid JSON value (dict, list, str, int, float, bool, None).
            path:  JSON Pointer path to this node. Defaults to "" (root).

        Returns:
            A TreeNode tree rooted at the appropriate node type.

        Raises:
            TypeError: If value is not a valid JSON type, or a dict contains
                a non-string key (JSON object keys must be strings).
            ValueError: If a cyclic container reference is detected.
        """
        return self._build(value, path, visited=set())

    def _build(
        self,
        value: Any,
        path: str,
        visited: set[int],
    ) -> TreeNode:
        """Recursive entry point that threads the ``visited`` cycle-detection set."""
        # CRITICAL: bool MUST be checked before int — bool subclasses int in Python
        if isinstance(value, bool):
            label = "true" if value else "false"
            return TreeNode(
                node_type=NodeType.SCALAR, label=label, path=path, value=value
            )

        if isinstance(value, dict):
            obj_id = id(value)
            if obj_id in visited:
                msg = f"Cyclic structure detected at path: {path or '/'!r}"
                raise ValueError(msg)
            visited.add(obj_id)
            try:
                return self._build_object(value, path, visited)
            finally:
                visited.discard(obj_id)

        if isinstance(value, list):
            list_id = id(value)
            if list_id in visited:
                msg = f"Cyclic structure detected at path: {path or '/'!r}"
                raise ValueError(msg)
            visited.add(list_id)
            try:
                return self._build_array(value, path, visited)
            finally:
                visited.discard(list_id)

        if isinstance(value, (str, int, float)):
            return TreeNode(
                node_type=NodeType.SCALAR, label=str(value), path=path, value=value
            )

        if value is None:
            return TreeNode(
                node_type=NodeType.SCALAR, label="null", path=path, value=None
            )

        raise TypeError(f"Unsupported JSON value type: {type(value)!r}")

    def _build_object(
        self,
        obj: dict[Any, Any],
        path: str,
        visited: set[int],
    ) -> TreeNode:
        """Build an OBJECT node with KEY->value child pairs.

        Args:
            obj:     The JSON object (Python dict).
            path:    JSON Pointer path to this object node.
            visited: Set of ``id()`` values for containers currently on the
                     recursion stack (cycle detection).

        Returns:
            An OBJECT TreeNode whose children are KEY nodes.

        Raises:
            TypeError: If any key is not a string — JSON object keys must be strings.
        """
        object_node = TreeNode(node_type=NodeType.OBJECT, label="", path=path)

        for key, val in obj.items():
            # JSON object keys MUST be strings.  Python dicts allow any
            # hashable key, but treating ``{1: "a"}`` as JSON silently is a
            # latent bug — downstream regex normalisation crashes on
            # non-strings.  Reject explicitly with a descriptive error.
            if not isinstance(key, str):
                msg = (
                    f"JSON object keys must be strings; "
                    f"got {type(key).__name__!r} at path {path or '/'!r}"
                )
                raise TypeError(msg)
            key_path = f"{path}/{_escape_path_segment(key)}"
            key_node = TreeNode(
                node_type=NodeType.KEY,
                label=_normalizer.normalize(key),
                path=key_path,
                raw_label=key,
            )
            child = self._build(val, path=key_path, visited=visited)
            key_node.children.append(child)
            object_node.children.append(key_node)

        return object_node

    def _build_array(
        self,
        arr: list[Any],
        path: str,
        visited: set[int],
    ) -> TreeNode:
        """Build an ARRAY node with ELEMENT->value child pairs.

        Args:
            arr:     The JSON array (Python list).
            path:    JSON Pointer path to this array node.
            visited: Set of ``id()`` values for containers currently on the
                     recursion stack (cycle detection).

        Returns:
            An ARRAY TreeNode whose children are ELEMENT nodes indexed by position.
        """
        array_node = TreeNode(node_type=NodeType.ARRAY, label="", path=path)

        for idx, item in enumerate(arr):
            # Array indices are decimal integers per RFC 6901 — no escaping
            # is required, but stringify explicitly so the path and label
            # formatting stay consistent.
            idx_str = str(idx)
            elem_path = f"{path}/{idx_str}"
            elem_node = TreeNode(
                node_type=NodeType.ELEMENT,
                label=idx_str,
                path=elem_path,
            )
            child = self._build(item, path=elem_path, visited=visited)
            elem_node.children.append(child)
            array_node.children.append(elem_node)

        return array_node
