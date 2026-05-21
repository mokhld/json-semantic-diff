"""Unit tests for :class:`PersistentEmbeddingCache`.

These tests are skipped when the optional ``diskcache`` extra is not
installed.  Install with ``pip install json-semantic-diff[diskcache]``.

Coverage:

- Cold-cache cross-instance sharing (two instances, same dir)
- Per-batch dedupe
- Backend-id namespacing (no collision across models)
- ``similarity()`` delegation when the backend exposes it
- Cosine fallback clamps negative scores to 0.0
- Persistence across instance recreation (close + reopen)
- Clear ImportError when ``diskcache`` is unavailable
- dtype preservation on round-trip through disk
"""

from __future__ import annotations

import builtins
import importlib
import sys
from typing import Any

import numpy as np
import pytest

pytest.importorskip("diskcache", reason="diskcache extra not installed")

from json_semantic_diff.backends import StaticBackend
from json_semantic_diff.persistent_cache import PersistentEmbeddingCache

# ---------------------------------------------------------------------------
# Spy helpers
# ---------------------------------------------------------------------------


class _SpyBackend:
    """Backend that records every ``embed`` call and returns deterministic vectors.

    Each unique input string maps to a fixed 4-D float64 vector derived from
    ``hash(s)`` (mod a small space) so two cache instances return identical
    vectors for identical strings.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.calls: list[list[str]] = []
        if model_name is not None:
            self.model_name = model_name

    def embed(self, strings: list[str]) -> np.ndarray:
        self.calls.append(list(strings))
        rows = []
        for s in strings:
            # Stable per-string vector: char codes padded/truncated to 4 dims.
            codes = [float(ord(c)) for c in s[:4].ljust(4, "\x00")]
            rows.append(np.array(codes, dtype=np.float64))
        return np.stack(rows) if rows else np.zeros((0, 4), dtype=np.float64)


# ---------------------------------------------------------------------------
# 1. Cold cache: two instances at the same dir share state
# ---------------------------------------------------------------------------


def test_two_instances_same_dir_share_state(tmp_path: Any) -> None:
    backend_a = _SpyBackend()
    backend_b = _SpyBackend()

    cache_a = PersistentEmbeddingCache(backend_a, cache_dir=tmp_path)
    cache_a.embed(["hello"])
    assert backend_a.calls == [["hello"]]
    cache_a.close()

    # Different instance, same dir — must hit the disk-backed store.
    cache_b = PersistentEmbeddingCache(backend_b, cache_dir=tmp_path)
    vecs = cache_b.embed(["hello"])

    assert backend_b.calls == [], (
        f"backend_b should not have been called; got {backend_b.calls}"
    )
    assert vecs.shape == (1, 4)
    cache_b.close()


# ---------------------------------------------------------------------------
# 2. Dedupe: duplicates in a batch only hit the backend once
# ---------------------------------------------------------------------------


def test_dedupe_within_batch(tmp_path: Any) -> None:
    backend = _SpyBackend()
    cache = PersistentEmbeddingCache(backend, cache_dir=tmp_path)

    result = cache.embed(["x", "x", "y"])

    assert backend.calls == [["x", "y"]], (
        f"expected one backend call with deduped ['x', 'y'], got {backend.calls}"
    )
    # Order preserved including the duplicate.
    assert result.shape == (3, 4)
    assert np.array_equal(result[0], result[1])
    cache.close()


# ---------------------------------------------------------------------------
# 3. Backend-id namespacing: two backends, same dir, no collision
# ---------------------------------------------------------------------------


def test_backend_id_namespacing_prevents_collision(tmp_path: Any) -> None:
    backend_small = _SpyBackend()
    backend_large = _SpyBackend()

    cache_small = PersistentEmbeddingCache(
        backend_small, cache_dir=tmp_path, backend_id="model-small"
    )
    cache_large = PersistentEmbeddingCache(
        backend_large, cache_dir=tmp_path, backend_id="model-large"
    )

    cache_small.embed(["shared"])
    assert backend_small.calls == [["shared"]]

    # Different backend_id => must miss and call backend_large.
    cache_large.embed(["shared"])
    assert backend_large.calls == [["shared"]], (
        "different backend_id must not see cached entries from another id"
    )

    cache_small.close()
    cache_large.close()


# ---------------------------------------------------------------------------
# 4. similarity() delegation when backend exposes it
# ---------------------------------------------------------------------------


def test_similarity_delegates_to_backend(tmp_path: Any) -> None:
    backend = StaticBackend()
    calls: list[tuple[str, str]] = []
    original = backend.similarity

    def spy_similarity(a: str, b: str) -> float:
        calls.append((a, b))
        return original(a, b)

    backend.similarity = spy_similarity  # type: ignore[method-assign]

    cache = PersistentEmbeddingCache(backend, cache_dir=tmp_path)
    score = cache.similarity("user_name", "userName")

    assert calls == [("user_name", "userName")], (
        f"similarity() should delegate; got call log {calls}"
    )
    assert 0.0 <= score <= 1.0
    cache.close()


# ---------------------------------------------------------------------------
# 5. Cosine fallback clamps negative scores to 0.0 (parity with EmbeddingCache)
# ---------------------------------------------------------------------------


def test_cosine_fallback_clamps_negative_to_zero(tmp_path: Any) -> None:
    """Opposite vectors must score 0.0, not -1.0 — STED relies on ``cost = 1 - sim``."""

    class _OpposingBackend:
        def embed(self, strings: list[str]) -> np.ndarray:
            mapping = {
                "a": np.array([1.0, 0.0], dtype=np.float64),
                "b": np.array([-1.0, 0.0], dtype=np.float64),
            }
            return np.stack([mapping[s] for s in strings])

    backend: Any = _OpposingBackend()
    cache = PersistentEmbeddingCache(backend, cache_dir=tmp_path)

    assert cache.similarity("a", "b") == pytest.approx(0.0, abs=1e-6)
    cache.close()


# ---------------------------------------------------------------------------
# 6. Persistence across instance recreation
# ---------------------------------------------------------------------------


def test_persistence_across_recreation(tmp_path: Any) -> None:
    backend1 = _SpyBackend()
    cache1 = PersistentEmbeddingCache(backend1, cache_dir=tmp_path)
    cache1.embed(["persist_me"])
    assert backend1.calls == [["persist_me"]]
    cache1.close()
    del cache1

    # Brand-new instance, fresh backend; should not re-embed.
    backend2 = _SpyBackend()
    cache2 = PersistentEmbeddingCache(backend2, cache_dir=tmp_path)
    cache2.embed(["persist_me"])

    assert backend2.calls == [], (
        f"persisted entry should survive instance recreation; got {backend2.calls}"
    )
    cache2.close()


# ---------------------------------------------------------------------------
# 7. ImportError when diskcache is missing
# ---------------------------------------------------------------------------


def test_import_error_when_diskcache_missing(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulate a missing ``diskcache`` install and verify a friendly hint."""
    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "diskcache":
            raise ImportError("simulated missing diskcache")
        return real_import(name, globals, locals, fromlist, level)

    # Drop any cached import so the lazy import path actually executes.
    monkeypatch.delitem(sys.modules, "diskcache", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    # Force a fresh import of the module so the patched __import__ is used
    # when ``import diskcache`` runs inside __init__.
    import json_semantic_diff.persistent_cache as pc_module

    importlib.reload(pc_module)

    backend = _SpyBackend()
    with pytest.raises(ImportError) as excinfo:
        pc_module.PersistentEmbeddingCache(backend, cache_dir=tmp_path)

    msg = str(excinfo.value)
    assert "diskcache" in msg
    assert "pip install" in msg
    # Reload again under the real importer so subsequent tests in the same
    # session see the normal module state.
    monkeypatch.setattr(builtins, "__import__", real_import)
    importlib.reload(pc_module)


# ---------------------------------------------------------------------------
# 8. dtype preservation on round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dtype_preserved_round_trip(
    tmp_path: Any, dtype: type[np.floating[Any]]
) -> None:
    class _DtypeBackend:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def embed(self, strings: list[str]) -> np.ndarray:
            self.calls.append(list(strings))
            return np.stack([np.array([1.0, 2.0, 3.0], dtype=dtype) for _ in strings])

    backend = _DtypeBackend()
    cache = PersistentEmbeddingCache(backend, cache_dir=tmp_path)

    first = cache.embed(["only"])
    assert first.dtype == dtype

    # Second call must come from disk and preserve dtype identically.
    backend.calls.clear()
    second = cache.embed(["only"])
    assert backend.calls == []
    assert second.dtype == dtype
    assert np.array_equal(first, second)
    cache.close()
