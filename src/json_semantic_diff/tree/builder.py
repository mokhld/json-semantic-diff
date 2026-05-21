"""TreeBuilder: converts any valid JSON value into a typed TreeNode tree.

Uses an iterative explicit-stack DFS to convert JSON dicts, lists, and scalar
values into a tree of TreeNode objects.  An iterative walk avoids the Python
recursion limit so pathologically deep JSON (tens of thousands of levels)
builds without ``RecursionError`` — see audit finding H1.

- Object keys are normalised via :class:`KeyNormalizer`.
- Array elements are wrapped in ``ELEMENT`` nodes with index-based labels.
- Scalar values preserve their original Python type in the ``TreeNode.value``
  field.

JSON Pointer paths (RFC 6901) are built during traversal:

- Root is ``""`` (empty string).
- Each level appends ``"/{escaped_key_or_index}"``.
- Per RFC 6901 section 4, key segments are escaped: ``~`` becomes ``~0``
  and ``/`` becomes ``~1``.  The ``~`` substitution is applied first so
  that a literal ``~`` in a key does not eat the ``~`` produced by the
  ``/`` substitution.  Array indices are decimal integers and need no
  escaping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

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


class _BuildTask(NamedTuple):
    """Pending value to convert into a TreeNode.

    The result is appended to ``target`` once the value (and all of its
    descendants) has been materialised.  ``target`` is the ``children``
    list of an already-created parent KEY/ELEMENT node — or, at the very
    top of ``build()``, a single-slot list that receives the root node.
    """

    value: Any
    path: str
    target: list[TreeNode]


class _FinishTask(NamedTuple):
    """Marker that pops a container's id off the active-ancestor set.

    Pushed onto the work stack right BEFORE the container's child tasks
    so that — under LIFO ordering — the marker fires only after every
    descendant has been processed.  At that point the container is no
    longer "on the recursion stack" and may legitimately appear again as
    a shared sibling subtree without being flagged as a cycle (see
    ``TestCycleDetection.test_repeated_non_cyclic_subtree_does_not_raise``).
    """

    container_id: int


_WorkItem = _BuildTask | _FinishTask


@dataclass
class TreeBuilder:
    """Converts any valid JSON value into a typed TreeNode tree.

    Uses an iterative explicit-stack DFS — see module docstring for the
    rationale.  The dispatch order is critical: ``bool`` MUST be checked
    before ``int`` because ``bool`` is a subclass of ``int`` in Python
    (``isinstance(True, int)`` is True).

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
        tracks every dict / list currently "on the recursion stack" —
        with the iterative walk that means containers whose subtree has
        not yet been fully built.  A ``_FinishTask`` marker removes a
        container's id once all its descendants have been processed, so
        the same shared subtree can legitimately appear twice in a DAG
        without being misreported as a cycle.

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
        visited: set[int] = set()
        root_slot: list[TreeNode] = []
        stack: list[_WorkItem] = [_BuildTask(value, path, root_slot)]

        while stack:
            item = stack.pop()
            if isinstance(item, _FinishTask):
                visited.discard(item.container_id)
                continue
            self._process_build_task(item, visited, stack)

        # The only way ``root_slot`` stays empty is if the input was an
        # unsupported type — but that raises ``TypeError`` inside
        # ``_process_build_task`` before reaching here.
        return root_slot[0]

    def _process_build_task(
        self,
        task: _BuildTask,
        visited: set[int],
        stack: list[_WorkItem],
    ) -> None:
        """Materialise one node, mutating ``visited`` / ``stack`` as needed.

        Splits out from :meth:`build` so the main loop stays compact.
        Scalar/null/bool values produce a TreeNode and append it directly to
        ``task.target``.  Containers (dict/list) create the OBJECT/ARRAY node
        plus its KEY/ELEMENT wrappers up front (preserving iteration order),
        push a ``_FinishTask`` marker, then push one ``_BuildTask`` per
        wrapped child so the LIFO stack visits them in original order.
        """
        val = task.value
        p = task.path
        target = task.target

        # CRITICAL: bool MUST be checked before int — bool subclasses int in Python.
        if isinstance(val, bool):
            label = "true" if val else "false"
            target.append(
                TreeNode(node_type=NodeType.SCALAR, label=label, path=p, value=val)
            )
            return

        if isinstance(val, dict):
            obj_id = id(val)
            if obj_id in visited:
                msg = f"Cyclic structure detected at path: {p or '/'!r}"
                raise ValueError(msg)
            # Validate every key up-front so a non-string key raises BEFORE
            # we mutate ``visited`` — leaving the set in a consistent state
            # for any caller that catches the TypeError.
            for key in val:
                if not isinstance(key, str):
                    msg = (
                        f"JSON object keys must be strings; "
                        f"got {type(key).__name__!r} at path {p or '/'!r}"
                    )
                    raise TypeError(msg)
            visited.add(obj_id)
            object_node = TreeNode(node_type=NodeType.OBJECT, label="", path=p)
            target.append(object_node)
            # FINISH marker is pushed BEFORE child tasks so LIFO pops it
            # LAST — i.e. only after every descendant has been processed.
            stack.append(_FinishTask(obj_id))
            # Build all KEY nodes in original order so OBJECT.children
            # preserves iteration order.
            key_value_pairs: list[tuple[Any, TreeNode]] = []
            for key, sub_val in val.items():
                key_path = f"{p}/{_escape_path_segment(key)}"
                key_node = TreeNode(
                    node_type=NodeType.KEY,
                    label=_normalizer.normalize(key),
                    path=key_path,
                    raw_label=key,
                )
                object_node.children.append(key_node)
                key_value_pairs.append((sub_val, key_node))
            # Push value-build tasks in REVERSE so LIFO pop processes
            # them left-to-right.
            for sub_val, key_node in reversed(key_value_pairs):
                stack.append(_BuildTask(sub_val, key_node.path, key_node.children))
            return

        if isinstance(val, list):
            list_id = id(val)
            if list_id in visited:
                msg = f"Cyclic structure detected at path: {p or '/'!r}"
                raise ValueError(msg)
            visited.add(list_id)
            array_node = TreeNode(node_type=NodeType.ARRAY, label="", path=p)
            target.append(array_node)
            stack.append(_FinishTask(list_id))
            item_elem_pairs: list[tuple[Any, TreeNode]] = []
            for idx, item in enumerate(val):
                # Array indices are decimal integers per RFC 6901 — no
                # escaping required, but stringify explicitly so the path
                # and label formatting stay consistent.
                idx_str = str(idx)
                elem_path = f"{p}/{idx_str}"
                elem_node = TreeNode(
                    node_type=NodeType.ELEMENT, label=idx_str, path=elem_path
                )
                array_node.children.append(elem_node)
                item_elem_pairs.append((item, elem_node))
            for item, elem_node in reversed(item_elem_pairs):
                stack.append(_BuildTask(item, elem_node.path, elem_node.children))
            return

        if isinstance(val, (str, int, float)):
            target.append(
                TreeNode(node_type=NodeType.SCALAR, label=str(val), path=p, value=val)
            )
            return

        if val is None:
            target.append(
                TreeNode(node_type=NodeType.SCALAR, label="null", path=p, value=None)
            )
            return

        raise TypeError(f"Unsupported JSON value type: {type(val)!r}")
