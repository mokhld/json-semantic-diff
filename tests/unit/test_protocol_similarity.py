"""Tests for the H8 fix: EmbeddingBackend Protocol now requires similarity().

A user-defined class with only an ``embed`` method MUST NOT satisfy the
Protocol anymore — both ``embed`` and ``similarity`` are required.
"""

from __future__ import annotations

import numpy as np

from json_semantic_diff.backends import StaticBackend
from json_semantic_diff.protocols import EmbeddingBackend


class _EmbedOnlyBackend:
    """Only ``embed`` — no longer conformant after H8."""

    def embed(self, strings: list[str]) -> np.ndarray:
        return np.zeros((len(strings), 4), dtype=np.float64)


class _FullBackend:
    """Both ``embed`` and ``similarity`` — fully conformant."""

    def embed(self, strings: list[str]) -> np.ndarray:
        return np.zeros((len(strings), 4), dtype=np.float64)

    def similarity(self, a: str, b: str) -> float:
        return 1.0 if a == b else 0.0


def test_embed_only_backend_no_longer_conformant() -> None:
    """After H8 — embed() alone is not enough to satisfy the Protocol."""
    assert isinstance(_EmbedOnlyBackend(), EmbeddingBackend) is False


def test_full_backend_is_conformant() -> None:
    """A class with both embed() and similarity() satisfies the Protocol."""
    assert isinstance(_FullBackend(), EmbeddingBackend) is True


def test_static_backend_still_conformant() -> None:
    """StaticBackend defines both methods — still conformant."""
    assert isinstance(StaticBackend(), EmbeddingBackend) is True
