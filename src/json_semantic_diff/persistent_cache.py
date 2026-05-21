"""PersistentEmbeddingCache: disk-backed caching proxy for any EmbeddingBackend.

Wraps any ``EmbeddingBackend`` and stores embedding vectors on disk via the
`diskcache <https://github.com/grantjenks/python-diskcache>`_ library, so that
repeated runs of the same process — or runs from completely different processes
— can share previously-computed embeddings.

The primary use case is amortizing the cost of expensive, paid embedding
backends such as OpenAI ``text-embedding-3-*``.  An in-memory cache (see
:class:`json_semantic_diff.cache.EmbeddingCache`) dies with the process; for
batch jobs, CI runs, or CLI re-invocations on the same data, ``OpenAIBackend``
ends up re-billing for embeddings it has already produced.  This class fixes
that by persisting ``(backend_id, string) -> np.ndarray`` rows to disk.

The on-disk size is bounded by ``max_size_bytes`` (default 1 GB) and diskcache
performs LRU + size-based eviction automatically — nothing here is permanent.
Point ``cache_dir`` at a shared directory (e.g. an NFS mount or a per-team
S3-FUSE path) to share a cache across machines.

.. warning::

    The on-disk format is whatever ``diskcache`` decides to use and is **not**
    a stable wire format.  Treat this cache as a pure optimization, not as a
    persistence contract.  Upgrading ``diskcache``, swapping Python versions,
    or moving between OSes may invalidate existing entries — that is fine, the
    cache will simply miss and re-embed.

This module imports ``diskcache`` lazily inside ``__init__`` so the library
remains an *optional* dependency.  Install it with::

    pip install json-semantic-diff[diskcache]

Example::

    from json_semantic_diff.persistent_cache import PersistentEmbeddingCache
    from json_semantic_diff.backends import StaticBackend  # or OpenAIBackend, ...

    backend = StaticBackend()
    cache = PersistentEmbeddingCache(
        backend,
        cache_dir="~/.cache/json-semantic-diff/embeddings",
        max_size_bytes=2 * 1024**3,  # 2 GB
    )

    vecs = cache.embed(["user_name", "address"])
    # ... process exits ...
    # In a fresh process, pointing at the same cache_dir:
    vecs_again = cache.embed(["user_name", "address"])  # served from disk
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from json_semantic_diff.protocols import EmbeddingBackend


_DISKCACHE_INSTALL_HINT = (
    "diskcache is required for PersistentEmbeddingCache. "
    "Install it with: pip install 'json-semantic-diff[diskcache]' "
    "(or: pip install diskcache)"
)


class PersistentEmbeddingCache:
    """Disk-backed caching proxy around any ``EmbeddingBackend``.

    Structurally satisfies the ``EmbeddingBackend`` Protocol (no inheritance
    required).  Embedding vectors are stored on disk via the ``diskcache``
    library, which performs LRU eviction once ``max_size_bytes`` is exceeded.

    Each ``PersistentEmbeddingCache`` instance opens its own ``diskcache.Cache``
    handle, but multiple instances pointed at the same ``cache_dir`` safely
    share state — diskcache uses SQLite under the hood and is documented as
    process- and thread-safe.

    Args:
        backend: Any object satisfying the ``EmbeddingBackend`` Protocol (must
            expose ``embed(strings: list[str]) -> np.ndarray``).
        cache_dir: Filesystem path that will hold the diskcache shard files.
            Created if it does not already exist.  Use a shared path (e.g. an
            NFS mount) to share embeddings across machines or CI workers.
        max_size_bytes: Soft upper bound on the total on-disk size, in bytes.
            Defaults to 1 GB — large enough to comfortably hold ~650K
            1536-dimensional float64 OpenAI embeddings, small enough to fit on
            a developer laptop without surprise.  Diskcache evicts LRU entries
            when this limit is exceeded.
        backend_id: Namespace tag mixed into every cache key so that swapping
            the underlying model (e.g. ``text-embedding-3-small`` ->
            ``-3-large``) cannot return stale vectors of the wrong dimension.
            Defaults to ``backend.model_name`` if available, else
            ``"default"``.

    Raises:
        ImportError: If the optional ``diskcache`` dependency is not
            installed.  The message includes the install command.
    """

    def __init__(
        self,
        backend: EmbeddingBackend,
        cache_dir: str | Path,
        max_size_bytes: int = 1_000_000_000,
        backend_id: str | None = None,
    ) -> None:
        # Lazy import — keeps diskcache optional.  We catch ImportError
        # rather than ModuleNotFoundError so a broken install (partial
        # uninstall, mis-pickled .pth) still surfaces our friendly hint.
        try:
            import diskcache  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise ImportError(_DISKCACHE_INSTALL_HINT) from exc

        self._backend: Any = backend
        if backend_id is None:
            backend_id = getattr(backend, "model_name", None) or "default"
        self._backend_id: str = backend_id

        # Normalize the path so str / Path / "~" inputs all work.
        resolved = Path(cache_dir).expanduser()
        resolved.mkdir(parents=True, exist_ok=True)
        self._cache_dir: Path = resolved

        # diskcache.Cache is thread- and process-safe per its docs, but we
        # still wrap the read/write critical section in a lock so that the
        # dedupe + insert sequence behaves atomically from this instance's
        # point of view.
        self._cache: Any = diskcache.Cache(
            str(resolved), size_limit=int(max_size_bytes)
        )
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def backend_id(self) -> str:
        """The namespace tag used for cache keys."""
        return self._backend_id

    @property
    def cache_dir(self) -> Path:
        """The directory holding the on-disk cache shards."""
        return self._cache_dir

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _key(self, s: str) -> tuple[str, str]:
        """Build the ``(backend_id, string)`` cache key.

        diskcache pickles the tuple — fine for our purposes, and matches the
        keying scheme used by the in-memory :class:`EmbeddingCache`.
        """
        return (self._backend_id, s)

    # ------------------------------------------------------------------
    # EmbeddingBackend Protocol surface
    # ------------------------------------------------------------------

    def embed(self, strings: list[str]) -> np.ndarray:
        """Return embeddings for ``strings``; only uncached strings hit the backend.

        Maintains input order: row ``i`` of the returned array corresponds to
        ``strings[i]``.  Individual row vectors (shape ``(D,)``) are persisted
        to disk; the full ``(N, D)`` matrix is reconstructed via ``np.stack``
        on return.

        Duplicate inputs are deduplicated before the backend call (order
        preserved via :func:`dict.fromkeys` semantics) so the wrapped backend
        never bills twice for the same string within a single batch.

        Args:
            strings: List of strings to embed.  May be empty.

        Returns:
            Numpy array of shape ``(len(strings), D)``.  Dtype matches whatever
            the wrapped backend returned originally.
        """
        results: dict[str, np.ndarray] = {}
        uncached: list[str] = []
        seen: set[str] = set()

        with self._lock:
            for s in strings:
                if s in seen:
                    continue
                seen.add(s)
                key = self._key(s)
                # diskcache exposes a dict-like interface; use .get with a
                # sentinel so we can distinguish "missing" from "stored None"
                # (although we never store None).
                cached = self._cache.get(key, default=None)
                if cached is not None:
                    results[s] = cached
                else:
                    uncached.append(s)

        if uncached:
            embeddings: np.ndarray = self._backend.embed(uncached)
            with self._lock:
                for s, vec in zip(uncached, embeddings, strict=True):
                    # Store individual row vectors (shape (D,)) so cached and
                    # uncached paths produce the same shape downstream.
                    self._cache[self._key(s)] = vec
                    results[s] = vec

        return np.stack([results[s] for s in strings])

    def similarity(self, a: str, b: str) -> float:
        """Return similarity score between two strings.

        Mirrors :meth:`EmbeddingCache.similarity` so behavior is identical
        regardless of which cache wraps the backend:

        - If the wrapped backend exposes ``similarity()`` (e.g.
          :class:`StaticBackend` with Levenshtein), delegate directly.  This
          avoids degenerate cosine results with backends whose ``embed()``
          returns non-discriminative stub arrays.
        - Otherwise, embed both strings and return their cosine similarity
          clamped to ``[0.0, 1.0]``.  Clamping (rather than rescaling)
          preserves the property that orthogonal *and* opposite vectors both
          score ``0.0`` — important because downstream STED cost is
          ``1 - sim`` and negative similarity would push cost above ``1.0``.

        Args:
            a: First string.
            b: Second string.

        Returns:
            Float in ``[0.0, 1.0]`` representing semantic similarity.
        """
        if hasattr(self._backend, "similarity"):
            return float(self._backend.similarity(a, b))

        vecs = self.embed([a, b])
        dot = float(np.dot(vecs[0], vecs[1]))
        norm_a = float(np.linalg.norm(vecs[0]))
        norm_b = float(np.linalg.norm(vecs[1]))
        raw = dot / (norm_a * norm_b + 1e-9)
        return max(0.0, min(1.0, raw))

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying diskcache handle.

        Safe to call multiple times.  After ``close()``, further ``embed()``
        or ``similarity()`` calls will raise.  Most callers do not need this:
        diskcache closes its handles on garbage collection.
        """
        with self._lock:
            close = getattr(self._cache, "close", None)
            if close is not None:
                close()

    def __enter__(self) -> PersistentEmbeddingCache:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
