"""Tests for EmbeddingBackend Protocol conformance.

Verifies that:
- User-defined classes with a conformant ``embed`` method satisfy the Protocol.
- Classes without ``embed`` (or with wrong method names) do not satisfy it.
- StaticBackend satisfies the Protocol structurally without inheritance.
"""

from __future__ import annotations

import numpy as np

from json_semantic_diff.backends import StaticBackend
from json_semantic_diff.protocols import EmbeddingBackend


class _UserBackend:
    """Minimal user-defined backend conforming to EmbeddingBackend."""

    def embed(self, strings: list[str]) -> np.ndarray:
        return np.zeros((len(strings), 4), dtype=np.float64)

    def similarity(self, a: str, b: str) -> float:
        return 1.0 if a == b else 0.0


class _NoEmbedBackend:
    """Class with no embed method — should NOT satisfy Protocol."""

    def predict(self, strings: list[str]) -> list[float]:
        return [0.0] * len(strings)


class _WrongNameBackend:
    """Class with wrong method name — should NOT satisfy Protocol."""

    def embed_texts(self, strings: list[str]) -> np.ndarray:
        return np.zeros((len(strings), 4), dtype=np.float64)


# ---------------------------------------------------------------------------
# Positive conformance tests
# ---------------------------------------------------------------------------


def test_user_defined_backend_passes_isinstance():
    """User-defined class with correct embed() signature satisfies Protocol."""
    backend = _UserBackend()
    assert isinstance(backend, EmbeddingBackend) is True


def test_user_defined_backend_can_produce_embeddings():
    """User-defined backend returns array of correct shape."""
    backend = _UserBackend()
    result = backend.embed(["hello", "world"])
    assert result.shape == (2, 4)


def test_static_backend_satisfies_protocol():
    """StaticBackend structurally satisfies EmbeddingBackend Protocol."""
    assert isinstance(StaticBackend(), EmbeddingBackend) is True


def test_protocol_does_not_require_inheritance():
    """StaticBackend should NOT have EmbeddingBackend in its MRO."""
    assert EmbeddingBackend not in type(StaticBackend()).__mro__


# ---------------------------------------------------------------------------
# Negative conformance tests
# ---------------------------------------------------------------------------


def test_class_without_embed_fails_isinstance():
    """Class without embed method does not satisfy Protocol."""
    obj = _NoEmbedBackend()
    assert isinstance(obj, EmbeddingBackend) is False


def test_class_with_wrong_method_name_fails_isinstance():
    """Class with embed_texts (wrong name) does not satisfy Protocol."""
    obj = _WrongNameBackend()
    assert isinstance(obj, EmbeddingBackend) is False
