"""Integrations subpackage for json-semantic-diff.

Contains integration adapters for external frameworks:
- pytest plugin (auto-discovered via pytest11 entry point)
- LangSmith evaluator adapter (LangSmithEvaluator)
- Braintrust scorer adapter (BraintrustScorer)
- W&B Weave scorer adapter (WeaveScorer)

Adapters with optional SDK dependencies (LangSmith, Weave) are imported
lazily via PEP 562 module-level ``__getattr__`` — a missing SDK does not
prevent the package from loading.  BraintrustScorer has no SDK dependency
and is imported eagerly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# BraintrustScorer has no SDK dependency — always importable.
from json_semantic_diff.integrations._braintrust import BraintrustScorer

if TYPE_CHECKING:
    # Surface the optional adapters to type checkers and IDEs so completion,
    # hover, and ``from json_semantic_diff.integrations import
    # LangSmithEvaluator`` resolve statically — even when the SDK extras are
    # not installed in the current environment.
    from json_semantic_diff.integrations._langsmith import LangSmithEvaluator
    from json_semantic_diff.integrations._weave import WeaveScorer

# Static ``__all__`` — kept alphabetically sorted (RUF022 compliance).  Type
# checkers and IDEs see the full public surface, independent of which
# optional extras happen to be installed.
__all__ = ["BraintrustScorer", "LangSmithEvaluator", "WeaveScorer"]


# PEP 562 lazy attribute resolution.  Importing ``json_semantic_diff.integrations``
# stays cheap (no langsmith/weave import on package load) and the optional
# names resolve only when actually accessed.  If the underlying SDK is not
# installed, ``ModuleNotFoundError`` (a subclass of ``ImportError``)
# propagates naturally with a useful message.
def __getattr__(name: str) -> object:
    if name == "LangSmithEvaluator":
        from json_semantic_diff.integrations._langsmith import LangSmithEvaluator

        return LangSmithEvaluator
    if name == "WeaveScorer":
        from json_semantic_diff.integrations._weave import WeaveScorer

        return WeaveScorer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
