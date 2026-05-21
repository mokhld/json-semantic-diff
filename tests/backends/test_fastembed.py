"""Unit tests for FastEmbedBackend.

Skipped automatically on base installs (no fastembed) via ``pytest.importorskip``.
A module-scoped fixture avoids re-initialising the ONNX model for every test,
keeping the suite fast after the one-time warm-up cost.
"""

from __future__ import annotations

import pytest

fastembed = pytest.importorskip("fastembed", reason="fastembed extra not installed")

import numpy as np  # noqa: E402

from json_semantic_diff.backends.fastembed import FastEmbedBackend  # noqa: E402
from json_semantic_diff.protocols import EmbeddingBackend  # noqa: E402

# ---------------------------------------------------------------------------
# Module-scoped fixture — one ONNX warm-up for the whole module
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def backend() -> FastEmbedBackend:
    """Return a single FastEmbedBackend instance shared across all tests."""
    return FastEmbedBackend()


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_protocol_conformance(backend: FastEmbedBackend) -> None:
    """FastEmbedBackend satisfies the EmbeddingBackend Protocol structurally."""
    assert isinstance(backend, EmbeddingBackend)


# ---------------------------------------------------------------------------
# Shape and dtype tests
# ---------------------------------------------------------------------------


def test_embed_returns_correct_shape(backend: FastEmbedBackend) -> None:
    """embed(3 strings) returns shape (3, 384) float32."""
    result = backend.embed(["user_name", "address", "created_at"])
    assert result.shape == (3, 384), f"Expected (3, 384), got {result.shape}"
    assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"


def test_embed_single_string(backend: FastEmbedBackend) -> None:
    """embed(1 string) returns shape (1, 384)."""
    result = backend.embed(["user_name"])
    assert result.shape == (1, 384), f"Expected (1, 384), got {result.shape}"


def test_embed_empty_list(backend: FastEmbedBackend) -> None:
    """embed([]) returns shape (0, 384) float32 without calling the model."""
    result = backend.embed([])
    assert result.shape == (0, 384), f"Expected (0, 384), got {result.shape}"
    assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_embed_deterministic(backend: FastEmbedBackend) -> None:
    """Same input strings produce identical embedding vectors on two calls."""
    strings = ["user_name", "address"]
    result1 = backend.embed(strings)
    result2 = backend.embed(strings)
    assert np.allclose(result1, result2), "embed() is non-deterministic"


# ---------------------------------------------------------------------------
# Discrimination
# ---------------------------------------------------------------------------


def test_embed_different_strings_produce_different_vectors(
    backend: FastEmbedBackend,
) -> None:
    """'user_name' and 'address' must NOT produce identical embedding vectors."""
    result = backend.embed(["user_name", "address"])
    row0 = result[0]
    row1 = result[1]
    assert not np.allclose(row0, row1), (
        "Distinct strings produced identical embedding vectors"
    )


# ---------------------------------------------------------------------------
# Ordering preservation
# ---------------------------------------------------------------------------


def test_embed_preserves_input_order(backend: FastEmbedBackend) -> None:
    """Row order in the output matches string order in the input."""
    strings_fwd = ["alpha", "beta", "gamma"]
    strings_rev = ["gamma", "beta", "alpha"]

    result_fwd = backend.embed(strings_fwd)
    result_rev = backend.embed(strings_rev)

    # gamma from fwd[2] should equal gamma from rev[0], etc.
    assert np.allclose(result_fwd[2], result_rev[0]), "gamma row mismatch"
    assert np.allclose(result_fwd[1], result_rev[1]), "beta row mismatch"
    assert np.allclose(result_fwd[0], result_rev[2]), "alpha row mismatch"


# ---------------------------------------------------------------------------
# Threading parameter
# ---------------------------------------------------------------------------


def test_threads_parameter_accepted() -> None:
    """FastEmbedBackend(intra_op_num_threads=1) does not raise."""
    backend_single = FastEmbedBackend(intra_op_num_threads=1)
    result = backend_single.embed(["test"])
    assert result.shape == (1, 384)


# ---------------------------------------------------------------------------
# Custom model name
# ---------------------------------------------------------------------------


def test_custom_model_name() -> None:
    """FastEmbedBackend with a custom model name does not raise on embed."""
    backend_custom = FastEmbedBackend(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    result = backend_custom.embed(["hello world"])
    # all-MiniLM-L6-v2 also produces 384-dim vectors
    assert result.ndim == 2
    assert result.shape[0] == 1
    assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# ImportError message simulation
# ---------------------------------------------------------------------------


def test_import_error_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify ImportError message when fastembed is not installed.

    Uses monkeypatch to simulate the missing package even though fastembed IS
    installed in this environment (importorskip passed above).
    """
    import builtins

    real_import = builtins.__import__

    def mock_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "fastembed":
            raise ImportError("mocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    with pytest.raises(
        ImportError, match=r"pip install json-semantic-diff\[fastembed\]"
    ):
        FastEmbedBackend()


# ---------------------------------------------------------------------------
# No similarity method
# ---------------------------------------------------------------------------


def test_similarity_method_present_and_in_range() -> None:
    """FastEmbedBackend exposes similarity() per the EmbeddingBackend Protocol.

    The method delegates to embed() + clamped cosine; callers wrap in
    EmbeddingCache to avoid recomputing the same vectors.
    """
    assert hasattr(FastEmbedBackend, "similarity")
    backend = FastEmbedBackend()
    score_self = backend.similarity("user_name", "user_name")
    score_diff = backend.similarity("user_name", "completely_unrelated_text")
    assert 0.0 <= score_self <= 1.0
    assert 0.0 <= score_diff <= 1.0
    assert score_self >= score_diff
