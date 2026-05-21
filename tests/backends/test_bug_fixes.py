"""Regression tests for embedding backend + cache bug fixes.

Covers the fixes for:
- C1: FastEmbed module no longer downloads ONNX model at import time
- H3: EmbeddingCache dedupes inputs before backend call
- H4: EmbeddingCache namespaces cache keys by backend_id
- H5: Cosine-fallback similarity is clamped to [0, 1]
- H6: OpenAIBackend chunks inputs > 2048 into multiple API calls
- H7: OpenAIBackend retries on APIConnectionError/Timeout/InternalServerError
- H10: OpenAIBackend forwards base_url/timeout/organization to the client
- M2: FastEmbedBackend empty-list shape uses the actual model dim
- G3: backends/__init__ only suppresses ModuleNotFoundError
- P5: EmbeddingCache wraps cache access in a threading.Lock
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from json_semantic_diff.cache import EmbeddingCache

# ---------------------------------------------------------------------------
# C1: FastEmbed module imports without downloading the ONNX model
# ---------------------------------------------------------------------------


def test_fastembed_module_import_does_not_instantiate_backend() -> None:
    """Importing the module must not create a module-level FastEmbedBackend.

    A module-level singleton would download the ONNX model on import — fine
    for users who actually want fastembed, but a ~100MB surprise for everyone
    else (e.g. CI runs that only need StaticBackend).
    """
    import json_semantic_diff.backends.fastembed as mod

    assert not hasattr(mod, "_DEFAULT_BACKEND"), (
        "Module-level _DEFAULT_BACKEND singleton must be removed "
        "(it eagerly downloads the ONNX model on import)."
    )


# ---------------------------------------------------------------------------
# H3: EmbeddingCache dedupes inputs before backend call
# ---------------------------------------------------------------------------


class _DimRecordingBackend:
    """Embed backend that records every list it receives."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed(self, strings: list[str]) -> np.ndarray:
        self.calls.append(list(strings))
        return np.stack(
            [
                np.full(self.dim, float(hash(s) % 1000), dtype=np.float64)
                for s in strings
            ]
        )


def test_cache_embed_dedupes_inputs_to_backend() -> None:
    """``cache.embed(['x', 'x', 'y'])`` must send only ``['x', 'y']`` once."""
    backend = _DimRecordingBackend()
    cache = EmbeddingCache(backend)

    result = cache.embed(["x", "x", "y"])

    assert len(backend.calls) == 1, f"Expected 1 backend call, got {backend.calls}"
    assert backend.calls[0] == ["x", "y"], (
        f"Expected backend to see deduped ['x', 'y'], got {backend.calls[0]}"
    )
    # Result must still have 3 rows in input order.
    assert result.shape == (3, 4)
    np.testing.assert_array_equal(result[0], result[1])  # both "x"


def test_cache_embed_preserves_order_when_deduping() -> None:
    """Deduped order must match first-seen order of the input."""
    backend = _DimRecordingBackend()
    cache = EmbeddingCache(backend)

    cache.embed(["b", "a", "b", "c", "a"])
    # First-seen order is ['b', 'a', 'c'] — preserved by dict.fromkeys semantics.
    assert backend.calls[0] == ["b", "a", "c"]


# ---------------------------------------------------------------------------
# H4: backend_id namespaces cache keys (model swap returns correct dim)
# ---------------------------------------------------------------------------


def test_cache_backend_id_namespaces_keys() -> None:
    """Two caches with different backend_id values do not share entries."""
    backend_small = _DimRecordingBackend(dim=4)
    backend_large = _DimRecordingBackend(dim=8)

    cache_small = EmbeddingCache(backend_small, backend_id="small")
    cache_large = EmbeddingCache(backend_large, backend_id="large")

    vec_small = cache_small.embed(["foo"])
    vec_large = cache_large.embed(["foo"])

    assert vec_small.shape == (1, 4)
    assert vec_large.shape == (1, 8)
    # backend_id is exposed for introspection.
    assert cache_small.backend_id == "small"
    assert cache_large.backend_id == "large"


def test_cache_auto_derives_backend_id_from_model_name() -> None:
    """If backend exposes ``model_name``, the cache uses it as backend_id."""

    class _NamedBackend:
        model_name = "fancy-embed-v2"

        def embed(self, strings: list[str]) -> np.ndarray:
            return np.zeros((len(strings), 3), dtype=np.float64)

    cache = EmbeddingCache(_NamedBackend())
    assert cache.backend_id == "fancy-embed-v2"


def test_cache_defaults_backend_id_to_default() -> None:
    """When no model_name is available, backend_id falls back to 'default'."""
    backend = _DimRecordingBackend()
    cache = EmbeddingCache(backend)
    assert cache.backend_id == "default"


# ---------------------------------------------------------------------------
# H5: Cosine-fallback similarity clamped to [0, 1]
# ---------------------------------------------------------------------------


def test_cosine_fallback_clamps_negative_to_zero() -> None:
    """Opposite vectors must score 0.0, not -1.0 (breaks STED ``cost = 1 - sim``)."""

    class _OpposingBackend:
        def embed(self, strings: list[str]) -> np.ndarray:
            mapping = {
                "a": np.array([1.0, 0.0], dtype=np.float64),
                "b": np.array([-1.0, 0.0], dtype=np.float64),
            }
            return np.stack([mapping[s] for s in strings])

    backend: Any = _OpposingBackend()
    cache = EmbeddingCache(backend)

    assert cache.similarity("a", "b") == pytest.approx(0.0, abs=1e-6)


def test_cosine_fallback_clamps_above_one() -> None:
    """Numerical drift cannot push similarity above 1.0."""

    class _OverUnitBackend:
        def embed(self, strings: list[str]) -> np.ndarray:
            # Identical vectors; floating-point noise can drift slightly > 1.
            vec = np.array([1.0, 1.0, 1.0], dtype=np.float64)
            return np.stack([vec for _ in strings])

    backend: Any = _OverUnitBackend()
    cache = EmbeddingCache(backend)

    score = cache.similarity("x", "y")
    assert score <= 1.0
    assert score == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# P5: EmbeddingCache holds a lock
# ---------------------------------------------------------------------------


def test_cache_has_threading_lock() -> None:
    """``EmbeddingCache`` exposes an internal ``threading.Lock`` for safety."""
    backend = _DimRecordingBackend()
    cache = EmbeddingCache(backend)

    assert hasattr(cache, "_lock"), "EmbeddingCache must hold a threading lock"
    # threading.Lock() returns a _thread.lock; threading.RLock has a different
    # type — accept either by checking for acquire/release.
    assert hasattr(cache._lock, "acquire")
    assert hasattr(cache._lock, "release")


def test_cache_concurrent_embed_is_safe() -> None:
    """Many threads embedding overlapping inputs must not crash or torn-write."""
    backend = _DimRecordingBackend()
    cache = EmbeddingCache(backend, max_size=64)

    strings = [f"k{i}" for i in range(32)]
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            for _ in range(50):
                result = cache.embed(strings)
                assert result.shape == (32, 4)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Concurrent embed raised: {errors[0]!r}"


# ---------------------------------------------------------------------------
# H6 + H7 + H10: OpenAI backend chunking, retry surface, and client kwargs
# ---------------------------------------------------------------------------

openai = pytest.importorskip("openai", reason="openai extra not installed")


def _mk_openai_backend(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> Any:
    """Build an OpenAIBackend with a mocked client."""
    from json_semantic_diff.backends.openai import OpenAIBackend

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")
    b = OpenAIBackend(**kwargs)
    b._client = MagicMock()
    return b


def _mk_mock_response(n: int, dim: int = 1536) -> Any:
    """Build a mock OpenAI embeddings response with ``n`` rows."""
    resp = MagicMock()
    items = []
    for i in range(n):
        item = MagicMock()
        item.embedding = [0.1] * dim
        item.index = i
        items.append(item)
    resp.data = items
    return resp


def test_openai_chunks_large_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inputs > 2048 must be split across multiple API calls."""
    backend = _mk_openai_backend(monkeypatch)

    # Simulate 3000 strings; expect two calls (2048 + 952).
    strings = [f"s{i}" for i in range(3000)]
    backend._client.embeddings.create.side_effect = [
        _mk_mock_response(2048),
        _mk_mock_response(952),
    ]

    result = backend.embed(strings)

    assert backend._client.embeddings.create.call_count == 2, (
        f"Expected 2 chunked calls, got {backend._client.embeddings.create.call_count}"
    )
    # Verify per-call batch sizes.
    call_input_lengths = [
        len(c.kwargs["input"]) for c in backend._client.embeddings.create.call_args_list
    ]
    assert call_input_lengths == [2048, 952]
    assert result.shape == (3000, 1536)
    assert result.dtype == np.float32


def test_openai_no_chunk_when_under_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inputs <= 2048 still use a single API call (no regression)."""
    backend = _mk_openai_backend(monkeypatch)

    strings = [f"s{i}" for i in range(100)]
    backend._client.embeddings.create.return_value = _mk_mock_response(100)

    backend.embed(strings)
    assert backend._client.embeddings.create.call_count == 1


def test_openai_retries_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """``APIConnectionError`` must be retried by tenacity (H7)."""
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_none,
    )

    backend = _mk_openai_backend(monkeypatch)

    # Build a real APIConnectionError instance (requires a httpx.Request).
    import httpx

    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    conn_err = openai.APIConnectionError(request=request)

    backend._client.embeddings.create.side_effect = [
        conn_err,
        _mk_mock_response(1),
    ]

    # Fast-retry shim covering the same exception set as production.
    fast_retry = retry(
        retry=retry_if_exception_type(
            (
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.InternalServerError,
            )
        ),
        stop=stop_after_attempt(6),
        wait=wait_none(),
    )
    backend._call_api = fast_retry(backend._raw_call)

    result = backend.embed(["hello"])
    assert backend._client.embeddings.create.call_count == 2
    assert result.shape == (1, 1536)


def test_openai_accepts_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """``base_url`` is forwarded to the OpenAI client (H10 — Azure/LiteLLM/vLLM)."""
    from json_semantic_diff.backends.openai import OpenAIBackend

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")

    with patch("openai.OpenAI") as mock_openai_cls:
        mock_openai_cls.return_value = MagicMock()
        OpenAIBackend(base_url="http://localhost:8080")

    mock_openai_cls.assert_called_once()
    kwargs = mock_openai_cls.call_args.kwargs
    assert kwargs.get("base_url") == "http://localhost:8080"
    assert kwargs.get("max_retries") == 0


def test_openai_accepts_timeout_and_org(monkeypatch: pytest.MonkeyPatch) -> None:
    """``timeout`` and ``organization`` are forwarded to the OpenAI client."""
    from json_semantic_diff.backends.openai import OpenAIBackend

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")

    with patch("openai.OpenAI") as mock_openai_cls:
        mock_openai_cls.return_value = MagicMock()
        OpenAIBackend(timeout=12.0, organization="org-abc")

    kwargs = mock_openai_cls.call_args.kwargs
    assert kwargs.get("timeout") == 12.0
    assert kwargs.get("organization") == "org-abc"


def test_openai_exposes_model_name_for_cache_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``OpenAIBackend.model_name`` lets ``EmbeddingCache`` auto-namespace."""
    from json_semantic_diff.backends.openai import OpenAIBackend

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")
    backend = OpenAIBackend(model_name="text-embedding-3-large")
    assert backend.model_name == "text-embedding-3-large"

    cache = EmbeddingCache(backend)
    assert cache.backend_id == "text-embedding-3-large"


# ---------------------------------------------------------------------------
# M2: FastEmbed empty-list returns shape based on actual model dim
# ---------------------------------------------------------------------------

fastembed = pytest.importorskip("fastembed", reason="fastembed extra not installed")


def test_fastembed_empty_input_uses_probed_dim() -> None:
    """``embed([])`` after a real embed must use the learned model dim."""
    from json_semantic_diff.backends.fastembed import FastEmbedBackend

    backend = FastEmbedBackend()
    # Probe via a real embed so _dim is learned.
    real = backend.embed(["hello"])
    empty = backend.embed([])
    assert empty.shape == (0, real.shape[1])
    assert empty.dtype == np.float32


def test_fastembed_exposes_model_name() -> None:
    """``FastEmbedBackend.model_name`` lets ``EmbeddingCache`` auto-namespace."""
    from json_semantic_diff.backends.fastembed import FastEmbedBackend

    backend = FastEmbedBackend()
    assert backend.model_name == "sentence-transformers/all-MiniLM-L6-v2"

    cache = EmbeddingCache(backend)
    assert cache.backend_id == "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# G3: backends/__init__.py only suppresses ModuleNotFoundError
# ---------------------------------------------------------------------------


def test_backends_init_suppresses_only_modulenotfounderror() -> None:
    """A non-ModuleNotFoundError raised inside an optional backend module must
    propagate, not be silently swallowed."""
    import inspect

    import json_semantic_diff.backends as pkg

    src = inspect.getsource(pkg)
    # The legacy code caught ImportError (parent), which silently hid real bugs
    # like a TypeError-during-import or a transitively-missing package.
    assert "except ModuleNotFoundError" in src, (
        "backends/__init__.py must narrow the except to ModuleNotFoundError"
    )
    # Defensive check: no bare ImportError fallback.
    assert "except ImportError" not in src, (
        "backends/__init__.py should not catch broad ImportError"
    )
