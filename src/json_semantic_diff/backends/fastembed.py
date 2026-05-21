"""FastEmbedBackend: ONNX embedding backend via the fastembed library.

Wraps ``fastembed.TextEmbedding`` with a lazy import so that the base install
(no fastembed installed) never triggers an ``ImportError`` at module level.
The ``fastembed`` package is only required when ``FastEmbedBackend`` is
*instantiated*.

Install the optional dependency with::

    pip install json-semantic-diff[fastembed]

Example::

    from json_semantic_diff.backends.fastembed import FastEmbedBackend

    backend = FastEmbedBackend()
    vecs = backend.embed(["user_name", "address"])
    print(vecs.shape)   # (2, 384)
    print(vecs.dtype)   # float32

**Offline / air-gapped workflow:**

On first run (online), pass ``cache_dir`` to persist the downloaded ONNX
model on disk::

    backend = FastEmbedBackend(cache_dir="./models")  # downloads once

On subsequent runs (offline / air-gapped), reuse the cache without
contacting HuggingFace::

    backend = FastEmbedBackend(cache_dir="./models", local_files_only=True)

**Model selection note:**
``BAAI/bge-small-en-v1.5`` was evaluated against a key-name discrimination
benchmark and produced a gap of ~0.16 (below the 0.25 threshold).
``sentence-transformers/all-MiniLM-L6-v2`` was substituted as the default and
produces a gap of ~0.29 with all related naming-convention pairs scoring 1.0
and all unrelated pairs scoring below 0.72 through the full STED stack.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any

import numpy as np


class FastEmbedBackend:
    """ONNX embedding backend wrapping ``fastembed.TextEmbedding``.

    Performs a lazy import of ``fastembed`` inside ``__init__``, so importing
    this module on a base install (no fastembed) does not raise
    ``ImportError``.  The error is deferred until the class is *instantiated*.

    Args:
        model_name: HuggingFace model identifier supported by fastembed.
            Defaults to ``"sentence-transformers/all-MiniLM-L6-v2"`` (384-dim,
            ONNX-optimized, Apache-2.0).  This was selected over
            ``"BAAI/bge-small-en-v1.5"`` after a discrimination benchmark
            found bge-small-en-v1.5 failed the gap threshold.
        intra_op_num_threads: Number of ONNX Runtime intra-op threads.
            Pass ``1`` to prevent thread explosion in multi-worker environments.
            ``None`` (default) lets fastembed/ONNX choose automatically.
        cache_dir: Directory to cache (download) or load the ONNX model from.
            Forwarded to ``fastembed.TextEmbedding(cache_dir=...)``.  When
            ``None`` (default), fastembed uses its built-in cache location
            (typically ``~/.cache/fastembed``).
        local_files_only: If True, never download from HuggingFace; only load
            from ``cache_dir`` (or the default fastembed cache).  When the
            installed ``fastembed.TextEmbedding`` accepts ``local_files_only``
            natively (it does for the versions this library targets), the
            kwarg is forwarded directly.  If a future fastembed release drops
            it, this backend falls back to setting ``HF_HUB_OFFLINE=1`` for
            the duration of the constructor call.

    Raises:
        ImportError: If ``fastembed`` is not installed.  The message includes
            the install command.
        ValueError: If ``local_files_only=True`` and the model cannot be loaded
            from the cache.  The message names the missing model and the cache
            directory that was searched.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        intra_op_num_threads: int | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
    ) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise ImportError(
                "fastembed is required for FastEmbedBackend. "
                "Install it with: pip install json-semantic-diff[fastembed]"
            ) from exc

        cache_dir_str: str | None = str(cache_dir) if cache_dir is not None else None

        # Build kwargs forwarded to TextEmbedding.  cache_dir is in the public
        # signature; local_files_only is passed via **kwargs (or emulated via
        # HF_HUB_OFFLINE if a future fastembed drops the kwarg).
        te_kwargs: dict[str, Any] = {
            "model_name": model_name,
            "threads": intra_op_num_threads,
            "cache_dir": cache_dir_str,
        }

        # Detect whether the installed TextEmbedding consumes local_files_only.
        # The current fastembed accepts it via **kwargs at every layer; we
        # detect by checking for an explicit param or by the presence of
        # **kwargs in the signature.
        sig = inspect.signature(TextEmbedding.__init__)
        params = sig.parameters
        supports_native = "local_files_only" in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )

        prev_offline: str | None = None
        try:
            if local_files_only:
                if supports_native:
                    te_kwargs["local_files_only"] = True
                else:
                    # Emulate via env var for the constructor call.
                    prev_offline = os.environ.get("HF_HUB_OFFLINE")
                    os.environ["HF_HUB_OFFLINE"] = "1"

            try:
                # Use Any annotation to avoid NameError when fastembed is
                # absent at type-check time (fastembed has no bundled stub).
                self._model: Any = TextEmbedding(**te_kwargs)
            except Exception as exc:
                if local_files_only:
                    where = cache_dir_str or "the default fastembed cache"
                    raise ValueError(
                        f"Model {model_name!r} not found in cache "
                        f"{where!r}; set local_files_only=False to download, "
                        "or pre-cache the model."
                    ) from exc
                raise
        finally:
            if local_files_only and not supports_native:
                if prev_offline is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = prev_offline

        self._model_name = model_name
        # Embedding dim is learned lazily on first embed() so we don't
        # hardcode 384 for non-MiniLM models.
        self._dim: int | None = None

    @property
    def model_name(self) -> str:
        """Return the underlying model identifier (used for cache namespacing)."""
        return self._model_name

    def embed(self, strings: list[str]) -> np.ndarray:
        """Return embeddings for ``strings`` as a float32 ndarray.

        FastEmbed's ``TextEmbedding.embed()`` returns a generator of
        ``(D,)`` float32 arrays - this method materialises the generator and
        stacks the rows into an ``(N, D)`` matrix.

        The embedding dimension is learned lazily on the first non-empty
        ``embed()`` call and cached on the instance.  Callers passing an empty
        list before any real embed will trigger a single throwaway probe to
        learn the dim so the returned ``(0, D)`` shape is correct for any
        model (not just the historical 384-dim default).

        Args:
            strings: Input strings to embed.  May be empty.

        Returns:
            Shape ``(N, D)`` numpy array with ``dtype=float32`` where
            ``N = len(strings)`` and ``D`` is the model's embedding dimension
            (e.g. 384 for ``all-MiniLM-L6-v2`` or 1024 for larger models).
        """
        if not strings:
            if self._dim is None:
                # Probe with a single throwaway embed to learn the model dim.
                probe = list(self._model.embed(["_"]))
                self._dim = int(probe[0].shape[0])
            return np.empty((0, self._dim), dtype=np.float32)

        # embed() returns a generator of (D,) float32 arrays - must materialise.
        vectors = list(self._model.embed(strings))
        stacked = np.stack(vectors).astype(np.float32)
        if self._dim is None:
            self._dim = int(stacked.shape[1])
        return stacked

    def similarity(self, a: str, b: str) -> float:
        """Cosine similarity of the two strings' embeddings, clamped to [0, 1].

        Each call embeds both strings - wrap in
        :class:`json_semantic_diff.cache.EmbeddingCache` for repeated lookups
        to avoid recomputing the same vectors.
        """
        vecs = self.embed([a, b])
        dot = float(np.dot(vecs[0], vecs[1]))
        denom = float(np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1]) + 1e-9)
        return max(0.0, min(1.0, dot / denom))
