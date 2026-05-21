"""ComparisonResult dataclass for semantic comparison output.

This module provides the rich result type returned by compare() calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ComparisonResult", "NodeContribution"]


@dataclass(frozen=True, slots=True)
class NodeContribution:
    """A single (path, contribution) record used by explain mode.

    Attributes:
        path: JSON Pointer to the affected subtree (e.g. ``/users/0/name``).
        contribution: Raw distance contribution in ``[0, +inf)``; higher
            values mean this node drove the final similarity score down
            more.  Distances are reported on the raw (un-normalised) STED
            scale so callers can sum them, sort them, or threshold them
            without re-deriving the normalisation chain.
        kind: One of ``"matched"`` (matched pair with non-zero distance),
            ``"unmatched_left"`` (present only on the left side),
            ``"unmatched_right"`` (present only on the right side), or
            ``"value_mismatch"`` (matched container/key whose underlying
            scalar value differs).
        detail: Optional human-readable note (e.g. ``"label 'uid' matched
            'user_id' via alias"``).  Defaults to an empty string.
    """

    path: str
    contribution: float
    kind: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Rich result of a compare() call.

    Attributes:
        similarity_score: Normalised similarity in [0.0, 1.0].  1.0 is identical.
        matched_pairs: Tuple of (left_path, right_path) JSON Pointer pairs for
            KEY nodes that were matched across the two documents.  Immutable
            so the audit trail cannot be mutated after construction.
        key_mappings: Mapping from raw left key name to raw right key name for
            each matched KEY pair.
        unmatched_left: JSON Pointer paths of KEY nodes present in the left
            document that had no counterpart in the right document.
        unmatched_right: JSON Pointer paths of KEY nodes present in the right
            document that had no counterpart in the left document.
        computation_time_ms: Wall-clock duration of the comparison in milliseconds.
        explanation: Optional tuple of per-path :class:`NodeContribution`
            records (highest impact first) explaining which parts of the
            documents drove the score.  Populated when
            ``STEDConfig.collect_explanation`` is True; defaults to the
            empty tuple otherwise, preserving backward compatibility.
    """

    similarity_score: float
    matched_pairs: tuple[tuple[str, str], ...]
    key_mappings: dict[str, str]
    unmatched_left: tuple[str, ...]
    unmatched_right: tuple[str, ...]
    computation_time_ms: float
    explanation: tuple[NodeContribution, ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict mirror of this result.

        Tuples are converted to lists so the dict round-trips cleanly through
        ``json.dumps``.  Field order matches the dataclass declaration.

        The ``explanation`` key is included only when at least one
        :class:`NodeContribution` is present, so callers who do not enable
        explain mode see the same payload as before.

        Returns:
            A plain ``dict`` with one entry per dataclass field; no nested
            tuples, dataclasses, or other JSON-incompatible types remain.
        """
        out: dict[str, Any] = {
            "similarity_score": self.similarity_score,
            "matched_pairs": [list(pair) for pair in self.matched_pairs],
            "key_mappings": dict(self.key_mappings),
            "unmatched_left": list(self.unmatched_left),
            "unmatched_right": list(self.unmatched_right),
            "computation_time_ms": self.computation_time_ms,
        }
        if self.explanation:
            out["explanation"] = [
                {
                    "path": c.path,
                    "contribution": c.contribution,
                    "kind": c.kind,
                    "detail": c.detail,
                }
                for c in self.explanation
            ]
        return out

    def to_json(self, indent: int | None = 2) -> str:
        """Serialise this result to a JSON string via :func:`json.dumps`.

        Args:
            indent: Passed through to ``json.dumps``.  ``None`` produces the
                most compact form; the default ``2`` produces human-readable
                output.

        Returns:
            A JSON-encoded string.
        """
        return json.dumps(self.to_dict(), indent=indent)
