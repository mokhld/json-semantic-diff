"""Backends subpackage for json-semantic-diff.

The base install provides only ``StaticBackend`` — a zero-ML-dependency
Levenshtein-based backend.  Optional backends (FastEmbed, OpenAI) are
available via extras:

    pip install json-semantic-diff[fastembed]   # FastEmbed ONNX backend
    pip install json-semantic-diff[openai]      # OpenAI embeddings backend

All backends satisfy the ``EmbeddingBackend`` Protocol structurally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from json_semantic_diff.backends.static import StaticBackend

if TYPE_CHECKING:
    # Surface the optional backends to type checkers and IDEs so completion,
    # hover, and ``from json_semantic_diff.backends import FastEmbedBackend``
    # resolve statically — even when the extras are not installed in the
    # current environment.
    from json_semantic_diff.backends.fastembed import FastEmbedBackend
    from json_semantic_diff.backends.openai import OpenAIBackend

# Static ``__all__`` — kept alphabetically sorted (RUF022 compliance).  Type
# checkers and IDEs see the full public surface, independent of which
# optional extras happen to be installed.
__all__ = ["FastEmbedBackend", "OpenAIBackend", "StaticBackend"]


# PEP 562 lazy attribute resolution.  Importing ``json_semantic_diff.backends``
# stays cheap (no fastembed/openai import on package load) and the optional
# names resolve only when actually accessed.  If the underlying module is not
# installed, ``ModuleNotFoundError`` propagates naturally with a useful
# message; any other ``ImportError`` (e.g. broken transitive dep) also
# propagates instead of being silently swallowed.
def __getattr__(name: str) -> object:
    if name == "FastEmbedBackend":
        from json_semantic_diff.backends.fastembed import FastEmbedBackend

        return FastEmbedBackend
    if name == "OpenAIBackend":
        from json_semantic_diff.backends.openai import OpenAIBackend

        return OpenAIBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
