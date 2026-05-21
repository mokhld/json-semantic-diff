"""OpenAIBackend: Cloud embedding backend via the OpenAI embeddings API.

Wraps ``openai.OpenAI`` with a lazy import so that the base install
(no openai/tenacity installed) never triggers an ``ImportError`` at module
level.  The ``openai`` and ``tenacity`` packages are only required when
``OpenAIBackend`` is *instantiated*.

The API key is read exclusively from the ``OPENAI_API_KEY`` environment
variable.  It is never accepted as a constructor parameter and never appears
in ``repr()``, ``str()``, or log output.

Rate-limited requests (HTTP 429) are retried automatically with jittered
exponential backoff via ``tenacity``.  The underlying ``openai.OpenAI`` client
is created with ``max_retries=0`` to prevent double-retry.

Install the optional dependency with::

    pip install json-semantic-diff[openai]

Example::

    from json_semantic_diff.backends.openai import OpenAIBackend

    backend = OpenAIBackend()
    vecs = backend.embed(["user_name", "address"])
    print(vecs.shape)   # (2, 1536)
    print(vecs.dtype)   # float32
"""

from __future__ import annotations

from typing import Any

import numpy as np

# OpenAI embeddings API hard limit on inputs per request (as of SDK v1.x).
# Requests larger than this are 400'd by the server.  We chunk client-side
# to keep callers from having to think about it.
_OPENAI_MAX_BATCH_SIZE = 2048


class OpenAIBackend:
    """OpenAI embedding backend using ``text-embedding-3-small``.

    Reads the API key exclusively from the ``OPENAI_API_KEY`` environment
    variable.  The key never appears in ``repr()``, ``str()``, or log output.

    Performs a lazy import of ``openai`` and ``tenacity`` inside ``__init__``,
    so importing this module on a base install does not raise
    ``ImportError``.  The error is deferred until the class is *instantiated*.

    Rate-limited API calls are retried with jittered exponential backoff via
    tenacity (up to 6 attempts).  The underlying ``openai.OpenAI`` client is
    created with ``max_retries=0`` to prevent double-retry.

    Args:
        model_name: OpenAI embedding model identifier.  Defaults to
            ``"text-embedding-3-small"`` (1536-dim, best quality/cost trade-off
            per OpenAI recommendation).
        base_url: Optional override for the API endpoint.  Forwarded to the
            ``OpenAI`` client — enables Azure OpenAI, LiteLLM, vLLM, or any
            OpenAI-compatible server.  ``None`` uses the SDK default
            (``https://api.openai.com/v1``).
        timeout: Optional per-request timeout in seconds.  Forwarded to the
            ``OpenAI`` client.  ``None`` uses the SDK default.
        organization: Optional ``OpenAI-Organization`` header value.
            Forwarded to the ``OpenAI`` client.  ``None`` uses the SDK
            default (reads ``OPENAI_ORG_ID`` env var if set).

    Raises:
        ImportError: If ``openai`` or ``tenacity`` is not installed.  The
            message includes the install command.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        base_url: str | None = None,
        timeout: float | None = None,
        organization: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai and tenacity are required for OpenAIBackend. "
                "Install with: pip install json-semantic-diff[openai]"
            ) from exc

        # Resolve retryable error classes.  Older SDKs (~1.0) may only expose
        # ``APIError`` / ``RateLimitError`` — fall back gracefully when the
        # narrower classes are missing.
        try:
            from openai import (
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
                RateLimitError,
            )

            retryable: tuple[type[BaseException], ...] = (
                RateLimitError,
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
            )
        except ImportError:
            # SDK too old for granular classes — retry on the superclass.
            from openai import APIError

            retryable = (APIError,)

        try:
            from tenacity import (
                retry,
                retry_if_exception_type,
                stop_after_attempt,
                wait_random_exponential,
            )
        except ImportError as exc:
            raise ImportError(
                "openai and tenacity are required for OpenAIBackend. "
                "Install with: pip install json-semantic-diff[openai]"
            ) from exc

        self._model_name = model_name
        # max_retries=0: tenacity is the sole retry controller.
        # The SDK default (max_retries=2) would create double-retry:
        # up to 6 (tenacity) * 3 (SDK) = 18 HTTP calls per embed().
        # Use Any annotation: openai is a lazy import, not available at
        # class definition time for type resolution.
        client_kwargs: dict[str, Any] = {"max_retries": 0}
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        if organization is not None:
            client_kwargs["organization"] = organization
        self._client: Any = OpenAI(**client_kwargs)

        # Build the retry decorator after error classes are in scope.
        # @retry cannot reference them at class definition time because
        # openai is not imported until __init__ runs.
        _retry = retry(
            retry=retry_if_exception_type(retryable),
            wait=wait_random_exponential(min=1, max=60),
            stop=stop_after_attempt(6),
        )
        # Wrap the raw API call with the retry decorator.
        self._call_api = _retry(self._raw_call)

    def __repr__(self) -> str:
        """Return a safe repr that never exposes the API key."""
        return f"OpenAIBackend(model={self._model_name!r})"

    @property
    def model_name(self) -> str:
        """Return the underlying model identifier (used for cache namespacing)."""
        return self._model_name

    def embed(self, strings: list[str]) -> np.ndarray:
        """Return embeddings for ``strings`` as a float32 (N, 1536) ndarray.

        Inputs larger than ``_OPENAI_MAX_BATCH_SIZE`` (2048) are chunked
        client-side and the resulting row blocks are concatenated.  This is
        a count-based chunk only — no per-token guard, since computing it
        would require a tokenizer dependency.

        Args:
            strings: Input strings to embed.  May be empty.

        Returns:
            Shape ``(N, 1536)`` numpy array with ``dtype=float32`` where
            ``N = len(strings)``.  Returns an empty ``(0, 1536)`` array
            without making any API calls when ``strings`` is empty.
        """
        if not strings:
            return np.empty((0, 1536), dtype=np.float32)

        if len(strings) <= _OPENAI_MAX_BATCH_SIZE:
            single: np.ndarray = self._call_api(strings)
            return single

        # Chunk by count to stay under the API's per-request input cap.
        chunks: list[np.ndarray] = []
        for i in range(0, len(strings), _OPENAI_MAX_BATCH_SIZE):
            batch = strings[i : i + _OPENAI_MAX_BATCH_SIZE]
            chunks.append(self._call_api(batch))
        return np.concatenate(chunks, axis=0)

    def similarity(self, a: str, b: str) -> float:
        """Cosine similarity of the two strings' embeddings, clamped to [0, 1].

        Each call issues an OpenAI API request — wrap in
        :class:`json_semantic_diff.cache.EmbeddingCache` for repeated lookups
        to avoid billing the same string twice.
        """
        vecs = self.embed([a, b])
        dot = float(np.dot(vecs[0], vecs[1]))
        denom = float(np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1]) + 1e-9)
        return max(0.0, min(1.0, dot / denom))

    def _raw_call(self, strings: list[str]) -> np.ndarray:
        """Make the raw embeddings API call — retried by tenacity via ``_call_api``.

        Args:
            strings: Non-empty list of strings to embed.

        Returns:
            Shape ``(N, 1536)`` numpy array with ``dtype=float32``.
        """
        response = self._client.embeddings.create(
            model=self._model_name,
            input=strings,
        )
        # Sort by index defensively — API guarantees input-order but
        # sorting prevents subtle bugs if that assumption ever changes.
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return np.array(
            [item.embedding for item in sorted_data],
            dtype=np.float32,
        )
