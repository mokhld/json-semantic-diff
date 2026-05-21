"""W&B Weave scorer adapter for json-semantic-diff.

Provides a factory function ``WeaveScorer`` that wraps an
``STEDComparator`` in the W&B Weave scorer interface (``weave.Scorer``
subclass pattern).

The ``weave.Scorer`` base class and ``@weave.op`` decorator are imported
lazily inside the factory function — base-install users who do not have
``weave`` installed will only see an ImportError when this factory is called,
not at module import time.

Install the optional SDK dependency with::

    pip install json-semantic-diff[weave]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from json_semantic_diff.comparator import STEDComparator

__all__ = ["WeaveScorer"]


def WeaveScorer(
    comparator: STEDComparator,
) -> Any:
    """Create a W&B Weave-compatible scorer instance.

    The returned object is a ``weave.Scorer`` subclass instance whose
    ``score()`` method is decorated with ``@weave.op`` for Weave tracking.

    The class is defined inside the factory to keep all ``weave`` imports
    lazy — only triggered when ``WeaveScorer()`` is called, not at module
    load time.

    On a successful comparison the scorer returns a dict with the scalar
    ``semantic_similarity`` plus the audit-trail fields ``key_mappings``,
    ``unmatched_left``, ``unmatched_right``, and ``computation_time_ms`` —
    Weave natively surfaces every key of the returned dict in the UI, so
    no JSON-encoding fallback is required here.

    N3 fix: when ``target`` is ``None`` the scorer returns
    ``semantic_similarity=None`` (with a ``skipped=True`` flag) rather than
    silently scoring ``0.0`` on missing reference data.

    Args:
        comparator: An ``STEDComparator`` instance to use for scoring.

    Returns:
        An instance of a ``weave.Scorer`` subclass with a ``score()`` method
        that returns a dict with ``semantic_similarity`` plus audit fields.

    Raises:
        ImportError: If ``weave`` is not installed.
    """
    try:
        import weave
        from weave import Scorer
    except ImportError as exc:
        raise ImportError(
            "weave is required: pip install json-semantic-diff[weave]"
        ) from exc

    class _WeaveSTEDScorer(Scorer):  # type: ignore[misc]
        @weave.op  # type: ignore[untyped-decorator]
        def score(self, output: Any, target: Any = None) -> dict[str, Any]:
            if target is None:
                # N3 fix: don't lie about the score on missing reference.
                return {
                    "semantic_similarity": None,
                    "skipped": True,
                    "reason": "target is None",
                }
            result = comparator.compare(output, target)
            return {
                "semantic_similarity": result.similarity_score,
                "key_mappings": dict(result.key_mappings),
                "unmatched_left": list(result.unmatched_left),
                "unmatched_right": list(result.unmatched_right),
                "computation_time_ms": result.computation_time_ms,
            }

    return _WeaveSTEDScorer()
