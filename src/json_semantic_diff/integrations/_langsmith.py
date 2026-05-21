"""LangSmith evaluator adapter for json-semantic-diff.

Provides a factory function ``LangSmithEvaluator`` that wraps an
``STEDComparator`` in the LangSmith evaluator interface (function-based
pattern).  The returned callable is compatible with ``langsmith.evaluate()``.

Install the optional SDK dependency with::

    pip install json-semantic-diff[langsmith]
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from json_semantic_diff.comparator import STEDComparator
    from json_semantic_diff.result import ComparisonResult

__all__ = ["LangSmithEvaluator"]


_SENTINEL: Any = object()


def _build_audit_comment(result: ComparisonResult) -> str:
    """Encode the audit-trail fields from ``result`` as a JSON string.

    LangSmith's ``EvaluationResult`` doesn't expose a structured ``metadata``
    field on every SDK version, but ``comment`` is a plain string field that
    is reliably surfaced in the LangSmith UI.  We pack the audit-trail fields
    into a JSON-serialised string so they round-trip through the platform
    without breaking the existing return-type contract.

    Args:
        result: A ``ComparisonResult`` produced by the wrapped comparator.

    Returns:
        A JSON-encoded string containing ``key_mappings``, ``unmatched_left``,
        ``unmatched_right``, and ``computation_time_ms``.
    """
    payload: dict[str, Any] = {
        "key_mappings": dict(result.key_mappings),
        "unmatched_left": list(result.unmatched_left),
        "unmatched_right": list(result.unmatched_right),
        "computation_time_ms": result.computation_time_ms,
    }
    return json.dumps(payload)


def LangSmithEvaluator(
    comparator: STEDComparator,
    output_key: str = "output",
) -> Any:
    """Create a LangSmith-compatible evaluator function.

    The returned callable follows the function-based LangSmith evaluator
    pattern (NOT the RunEvaluator ABC pattern) to avoid class-definition-time
    imports.  All LangSmith SDK imports are lazy — base-install users who do
    not have langsmith installed will only see an ImportError when this factory
    is called, not at module import time.

    Each ``EvaluationResult`` carries the scalar score plus a ``comment``
    field whose payload is a JSON object with ``key_mappings``,
    ``unmatched_left``, ``unmatched_right``, and ``computation_time_ms`` —
    the audit trail from the underlying ``ComparisonResult``.  The
    LangSmith ``EvaluationResult`` API doesn't have a stable structured
    ``metadata`` field across SDK versions, so we pack this audit trail into
    ``comment`` (a JSON string) for maximum compatibility.

    If ``run.outputs`` is ``None`` or doesn't contain ``output_key`` (and the
    same for ``example.outputs``), the evaluator returns ``score=None`` with
    a ``comment`` describing the skip reason — rather than silently scoring
    ``0.0`` on missing data.

    Args:
        comparator: An ``STEDComparator`` instance to use for scoring.
        output_key: Key to extract from ``run.outputs`` and ``example.outputs``.
            Defaults to ``"output"``.

    Returns:
        A callable ``_evaluator(run, example=None)`` that returns a LangSmith
        ``EvaluationResult`` with ``key="semantic_similarity"`` and a score in
        ``[0.0, 1.0]`` (or ``None`` when input data is missing).

    Raises:
        ImportError: If ``langsmith`` is not installed.
    """
    try:
        import langsmith  # noqa: F401
        from langsmith.evaluation.evaluator import EvaluationResult
    except ImportError as exc:
        raise ImportError(
            "langsmith is required: pip install json-semantic-diff[langsmith]"
        ) from exc

    def _evaluator(run: Any, example: Any = None) -> Any:
        # N2 fix: don't lie about the score when input data is missing.
        # If run.outputs is None or doesn't contain output_key, return
        # score=None rather than silently producing a 0.0 score.
        run_outputs: Any = getattr(run, "outputs", None)
        if run_outputs is None or output_key not in run_outputs:
            return EvaluationResult(
                key="semantic_similarity",
                score=None,
                comment=json.dumps(
                    {
                        "skipped": True,
                        "reason": (
                            f"run.outputs is None or missing key {output_key!r}"
                        ),
                    }
                ),
            )

        actual: Any = run_outputs.get(output_key, _SENTINEL)
        if actual is _SENTINEL:
            return EvaluationResult(
                key="semantic_similarity",
                score=None,
                comment=json.dumps(
                    {
                        "skipped": True,
                        "reason": f"run.outputs missing key {output_key!r}",
                    }
                ),
            )

        example_outputs: Any = (
            getattr(example, "outputs", None) if example is not None else None
        )
        if example_outputs is None or output_key not in example_outputs:
            return EvaluationResult(
                key="semantic_similarity",
                score=None,
                comment=json.dumps(
                    {
                        "skipped": True,
                        "reason": (
                            f"example.outputs is None or missing key {output_key!r}"
                        ),
                    }
                ),
            )

        expected: Any = example_outputs[output_key]
        result = comparator.compare(actual, expected)
        return EvaluationResult(
            key="semantic_similarity",
            score=result.similarity_score,
            comment=_build_audit_comment(result),
        )

    return _evaluator
