"""Braintrust scorer adapter for json-semantic-diff.

Provides a factory function ``BraintrustScorer`` that wraps an
``STEDComparator`` in the Braintrust scorer interface (plain function pattern).

No Braintrust SDK import is required — the scorer is a plain Python function
with the signature that Braintrust expects.  Install Braintrust separately
if you also need ``braintrust.Eval()`` runner support::

    pip install json-semantic-diff[braintrust]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from json_semantic_diff.comparator import STEDComparator

__all__ = ["BraintrustScore", "BraintrustScorer"]


class BraintrustScore(float):
    """A ``float`` carrying audit-trail metadata for the Braintrust UI.

    Subclasses ``float`` so the existing scorer contract (return value is
    treated as a float in [0.0, 1.0]) continues to hold: arithmetic,
    comparisons, and ``isinstance(x, float)`` checks all behave identically
    to a plain float.  The added ``metadata`` attribute carries the audit
    trail (key mappings, unmatched paths, computation time) that Braintrust
    surfaces alongside the score.

    ``__new__`` is overridden to capture the metadata dict, since ``float``
    is immutable and ``__init__`` runs after ``__new__`` returns.

    Attributes:
        metadata: The audit-trail dict.  Keys: ``key_mappings``,
            ``unmatched_left``, ``unmatched_right``, ``computation_time_ms``.
    """

    metadata: dict[str, Any]

    def __new__(cls, value: float, metadata: dict[str, Any]) -> BraintrustScore:
        instance = super().__new__(cls, value)
        instance.metadata = metadata
        return instance


def BraintrustScorer(
    comparator: STEDComparator,
) -> Any:
    """Create a Braintrust-compatible scorer function.

    The returned callable follows Braintrust's scorer interface.  It requires
    no Braintrust SDK imports — it is a plain function factory.

    On a successful comparison the scorer returns a :class:`BraintrustScore`
    — a ``float`` subclass in ``[0.0, 1.0]`` that also carries a
    ``metadata`` dict with ``key_mappings``, ``unmatched_left``,
    ``unmatched_right``, and ``computation_time_ms``.  Because
    ``BraintrustScore`` IS a ``float`` (subclass), the existing return-type
    contract is preserved — arithmetic and ``isinstance(x, float)`` continue
    to work exactly as before.  When ``expected`` is ``None`` the scorer
    still returns ``None`` (no score available).

    Args:
        comparator: An ``STEDComparator`` instance to use for scoring.

    Returns:
        A callable ``_scorer(input, output, expected=None, metadata=None)``
        that returns a :class:`BraintrustScore` (``float`` subclass) in
        ``[0.0, 1.0]`` or ``None`` when no ``expected`` is provided.  The
        function name is set to ``"semantic_similarity"`` for Braintrust
        display purposes.
    """

    def _scorer(
        input: Any,
        output: Any,
        expected: Any = None,
        metadata: Any = None,
    ) -> BraintrustScore | None:
        if expected is None:
            return None
        result = comparator.compare(output, expected)
        audit: dict[str, Any] = {
            "key_mappings": dict(result.key_mappings),
            "unmatched_left": list(result.unmatched_left),
            "unmatched_right": list(result.unmatched_right),
            "computation_time_ms": result.computation_time_ms,
        }
        return BraintrustScore(result.similarity_score, audit)

    _scorer.__name__ = "semantic_similarity"
    return _scorer
