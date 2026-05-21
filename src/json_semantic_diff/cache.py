"""EmbeddingCache: LRU-backed caching proxy for any EmbeddingBackend.

Wraps any EmbeddingBackend-conformant object and transparently caches
embedding results in memory. Cached strings bypass the backend on
subsequent ``embed()`` calls. LRU eviction occurs silently when
``max_size`` is exceeded — no error is raised.

Each ``EmbeddingCache`` instance maintains its own ``LRUCache`` — there is
no class-level shared state, so two separate instances never interfere
with each other.

Cache keys are namespaced by ``(backend_id, string)`` so that swapping the
underlying model (e.g. ``text-embedding-3-small`` to ``-3-large``) cannot
return stale vectors of the wrong dimension.  ``backend_id`` defaults to
``"default"`` for backward compatibility.

``cachetools.LRUCache`` is not thread-safe on its own; all reads and
writes go through a ``threading.Lock`` to prevent torn updates when
multiple threads hit the same cache.

Example::

    from json_semantic_diff.cache import EmbeddingCache
    from json_semantic_diff.backends import StaticBackend

    backend = StaticBackend()
    cache = EmbeddingCache(backend, max_size=512)

    # First call hits the backend
    vecs = cache.embed(["user_name", "address"])

    # Second call is fully served from memory — backend never called
    vecs_again = cache.embed(["user_name", "address"])
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import numpy as np
from cachetools import LRUCache

if TYPE_CHECKING:
    from json_semantic_diff.protocols import EmbeddingBackend


class EmbeddingCache:
    """LRU-backed caching proxy around any EmbeddingBackend.

    Satisfies the ``EmbeddingBackend`` Protocol structurally (no inheritance
    required). Each instance maintains its own ``LRUCache`` — no cross-instance
    sharing. LRU eviction is silent: the least-recently-used entry is dropped
    when ``max_size`` is exceeded.

    Args:
        backend: Any object satisfying the ``EmbeddingBackend`` Protocol
            (has an ``embed(strings: list[str]) -> np.ndarray`` method).
        max_size: Maximum number of string embeddings to hold in memory.
            Defaults to 512. When exceeded, the least-recently-used entry
            is silently evicted.
        backend_id: Namespace tag mixed into every cache key.  Pass a model
            identifier (e.g. ``"text-embedding-3-small"``) so two caches
            wrapping different models cannot collide on the same string.
            Defaults to ``"default"`` for backward compatibility.  If
            ``None``, attempts to read ``backend.model_name`` and falls back
            to ``"default"``.
    """

    def __init__(
        self,
        backend: EmbeddingBackend,
        max_size: int = 512,
        backend_id: str | None = None,
    ) -> None:
        # Store as Any at runtime — structural duck-typing, no Protocol coupling.
        self._backend: Any = backend
        # Auto-derive a backend_id from the wrapped backend when none is
        # supplied.  This lets FastEmbed/OpenAI backends namespace their own
        # caches without callers having to pass model_name through manually.
        if backend_id is None:
            backend_id = getattr(backend, "model_name", None) or "default"
        self._backend_id: str = backend_id
        # Cache key is (backend_id, string) so model swaps cannot return
        # wrong-dim vectors silently.
        self._cache: LRUCache[tuple[str, str], np.ndarray] = LRUCache(maxsize=max_size)
        # cachetools.LRUCache is not thread-safe — wrap all access in a lock.
        self._lock = threading.Lock()
        # Audit I2 (wave 8): pairwise similarity cache.  The algorithm
        # computes ``backend.similarity(label_a, label_b)`` for every KEY
        # pair inside Hungarian cost-matrix building, then the comparator
        # extracts key-mapping data with a second Hungarian pass over the
        # SAME label pairs.  Caching the scalar result here halves the
        # Levenshtein/cosine work on wide objects without altering any
        # downstream value.  Keyed canonically (smaller string first) so
        # ``similarity(a, b)`` and ``similarity(b, a)`` share an entry.
        self._sim_cache: LRUCache[tuple[str, str, str], float] = LRUCache(
            maxsize=max_size * max_size
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def max_size(self) -> int:
        """The maximum number of entries this cache can hold."""
        return int(self._cache.maxsize)

    @property
    def curr_size(self) -> int:
        """The current number of entries stored in the cache."""
        with self._lock:
            return int(self._cache.currsize)

    @property
    def backend_id(self) -> str:
        """The namespace tag used for cache keys."""
        return self._backend_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _key(self, s: str) -> tuple[str, str]:
        """Build the (backend_id, string) cache key."""
        return (self._backend_id, s)

    # ------------------------------------------------------------------
    # EmbeddingBackend Protocol surface
    # ------------------------------------------------------------------

    def embed(self, strings: list[str]) -> np.ndarray:
        """Return embeddings for ``strings``; only uncached strings hit the backend.

        Maintains input order: row ``i`` of the returned array corresponds to
        ``strings[i]``. Individual row vectors (shape ``(D,)``) are stored in
        the cache; the full ``(N, D)`` matrix is reconstructed via
        ``np.stack`` on return.

        Duplicate inputs are deduplicated before the backend call (order
        preserved via ``dict.fromkeys``) so the wrapped backend never bills
        twice for the same string within a single batch.

        Args:
            strings: List of strings to embed. May be empty.

        Returns:
            Shape ``(N, D)`` float64 numpy array where ``N = len(strings)``.
        """
        # Snapshot already-cached rows AND collect deduped uncached inputs in
        # a single locked pass.  We must hold the cached vectors locally so
        # that LRU eviction triggered by *new* inserts below cannot drop a
        # row we still need to return.
        results: dict[str, np.ndarray] = {}
        uncached: list[str] = []
        seen: set[str] = set()
        with self._lock:
            for s in strings:
                if s in seen:
                    continue
                seen.add(s)
                key = self._key(s)
                if key in self._cache:
                    results[s] = self._cache[key]
                else:
                    uncached.append(s)

        if uncached:
            embeddings: np.ndarray = self._backend.embed(uncached)
            with self._lock:
                for s, vec in zip(uncached, embeddings, strict=True):
                    # Store individual row vectors (shape (D,)) — not the
                    # full matrix.  This prevents shape inconsistency between
                    # cached and uncached paths.
                    self._cache[self._key(s)] = vec
                    results[s] = vec

        # Stack in input order (including duplicates) to produce a
        # consistent (N, D) shape.
        return np.stack([results[s] for s in strings])

    def similarity(self, a: str, b: str) -> float:
        """Return similarity score between two strings.

        If the wrapped backend exposes a ``similarity()`` method (e.g.
        ``StaticBackend`` with Levenshtein), delegates directly. This avoids
        the degenerate cosine issue with backends whose ``embed()`` returns
        non-discriminative representations (e.g. ``StaticBackend``'s ``(N,1)``
        stub arrays).

        For backends without ``similarity()`` (e.g. future ML backends in
        Phases 8/9), embeds both strings and returns their cosine similarity
        clamped to ``[0.0, 1.0]``.  Clamping (rather than rescaling) preserves
        the property that orthogonal and opposite vectors both score 0.0 —
        important because downstream cost is ``1 - sim`` and negative
        similarity would push cost above 1.0, violating STED invariants.

        Audit I2 (wave 8): the result is memoised under a canonical
        (smaller-string-first) key so the algorithm and the comparator's
        downstream key-extraction pass don't re-run Levenshtein/cosine
        on the same label pair.  This is a pure perf win — bit-identical
        scores, just fewer backend calls on wide objects.

        Args:
            a: First string.
            b: Second string.

        Returns:
            Float in [0.0, 1.0] representing semantic similarity.
        """
        # Canonical key: order-independent so (a, b) and (b, a) share a slot.
        # Static & embedding backends in this repo are symmetric on labels;
        # this assumption is documented on the EmbeddingBackend Protocol.
        sim_key = (self._backend_id, a, b) if a <= b else (self._backend_id, b, a)
        with self._lock:
            cached_sim = self._sim_cache.get(sim_key)
        if cached_sim is not None:
            return cached_sim

        if hasattr(self._backend, "similarity"):
            result = float(self._backend.similarity(a, b))
            with self._lock:
                self._sim_cache[sim_key] = result
            return result

        # Fallback: embed and compute cosine similarity, clamped to [0, 1].
        vecs = self.embed([a, b])
        dot = float(np.dot(vecs[0], vecs[1]))
        norm_a = float(np.linalg.norm(vecs[0]))
        norm_b = float(np.linalg.norm(vecs[1]))
        raw = dot / (norm_a * norm_b + 1e-9)
        clamped = max(0.0, min(1.0, raw))
        with self._lock:
            self._sim_cache[sim_key] = clamped
        return clamped
