"""Unit tests for OpenAIBackend.

Skipped automatically on base installs (no openai/tenacity) via
``pytest.importorskip``.  All tests use mocks -- no real OpenAI API calls are
made.  A function-scoped fixture is used because the OpenAI client is
lightweight (no ONNX loading), unlike FastEmbedBackend.

Test coverage is grouped by Phase 9 success criteria:
  SC1 -- Import/Install verification
  SC2 -- API key security (no leakage in repr/str)
  SC3 -- Rate-limit retry with exponential backoff
  SC4 -- ImportError with install hint when openai is missing
"""

from __future__ import annotations

import builtins
import importlib
import inspect
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

openai = pytest.importorskip("openai", reason="openai extra not installed")

import httpx  # noqa: E402

from json_semantic_diff.backends.openai import OpenAIBackend  # noqa: E402
from json_semantic_diff.protocols import EmbeddingBackend  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_mock_response(strings: list[str], dim: int = 1536) -> Any:
    """Build a mock object matching the OpenAI embeddings response shape.

    ``response.data`` is a list of mock items, each with
    ``.embedding = [0.1] * dim`` and ``.index = i``.
    """
    mock_response = MagicMock()
    items = []
    for i, _ in enumerate(strings):
        item = MagicMock()
        item.embedding = [0.1] * dim
        item.index = i
        items.append(item)
    mock_response.data = items
    return mock_response


def make_rate_limit_error() -> openai.RateLimitError:
    """Build an ``openai.RateLimitError`` instance using an httpx 429 response."""
    mock_response = httpx.Response(
        429,
        request=httpx.Request("POST", "https://api.openai.com/v1/embeddings"),
    )
    return openai.RateLimitError(
        "Rate limit exceeded",
        response=mock_response,
        body=None,
    )


def make_authentication_error() -> openai.AuthenticationError:
    """Build an ``openai.AuthenticationError`` instance using an httpx 401 response."""
    mock_response = httpx.Response(
        401,
        request=httpx.Request("POST", "https://api.openai.com/v1/embeddings"),
    )
    return openai.AuthenticationError(
        "Invalid API key",
        response=mock_response,
        body=None,
    )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> OpenAIBackend:
    """Return an OpenAIBackend with OPENAI_API_KEY set and client mocked.

    Function-scoped (default) -- the OpenAI client is lightweight, no ONNX
    loading.  Sets the env var so the SDK does not complain, then replaces
    ``_client`` with a ``MagicMock`` so no real API calls are made.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")
    b = OpenAIBackend()
    b._client = MagicMock()
    return b


# ---------------------------------------------------------------------------
# SC1 -- Import / Install (tests 1-3)
# ---------------------------------------------------------------------------


def test_import_from_backends_package() -> None:
    """``from json_semantic_diff.backends import OpenAIBackend`` succeeds (SC1)."""
    from json_semantic_diff.backends import OpenAIBackend as _OAB

    assert _OAB is OpenAIBackend


def test_protocol_conformance(backend: OpenAIBackend) -> None:
    """OpenAIBackend satisfies the EmbeddingBackend Protocol structurally (SC1)."""
    assert isinstance(backend, EmbeddingBackend)


def test_default_model_name(backend: OpenAIBackend) -> None:
    """Default model is ``text-embedding-3-small`` (SC1)."""
    assert "text-embedding-3-small" in repr(backend)


# ---------------------------------------------------------------------------
# SC2 -- API key security (tests 4-6)
# ---------------------------------------------------------------------------


def test_repr_does_not_contain_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``repr(backend)`` does not expose the API key (SC2)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-supersecret-key")
    b = OpenAIBackend()
    b._client = MagicMock()
    assert "sk-" not in repr(b), f"API key leaked in repr: {repr(b)!r}"


def test_str_does_not_contain_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``str(backend)`` does not expose the API key (SC2)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-supersecret-key")
    b = OpenAIBackend()
    b._client = MagicMock()
    assert "sk-" not in str(b), f"API key leaked in str: {str(b)!r}"


def test_no_api_key_parameter() -> None:
    """``OpenAIBackend.__init__`` has no ``api_key`` parameter (SC2).

    The API key is read exclusively from the ``OPENAI_API_KEY`` environment
    variable -- passing it as a parameter would risk accidental exposure in
    stack traces or argument logs.
    """
    sig = inspect.signature(OpenAIBackend.__init__)
    assert "api_key" not in sig.parameters, (
        f"api_key found in __init__ parameters: {list(sig.parameters.keys())}"
    )


# ---------------------------------------------------------------------------
# SC3 -- Rate-limit retry (tests 7-9)
# ---------------------------------------------------------------------------


def test_retry_on_rate_limit(backend: OpenAIBackend) -> None:
    """Raises ``RateLimitError`` twice then succeeds; assert called 3 times (SC3)."""
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_none,
    )

    strings = ["user_name", "address", "email"]
    mock_response = make_mock_response(strings)

    # Fail twice, then return success on the third call
    backend._client.embeddings.create.side_effect = [
        make_rate_limit_error(),
        make_rate_limit_error(),
        mock_response,
    ]

    # Replace _call_api with a fast-retry version (wait_none) to avoid
    # real delays in the test suite
    fast_retry = retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        stop=stop_after_attempt(6),
        wait=wait_none(),
    )
    backend._call_api = fast_retry(backend._raw_call)

    result = backend.embed(strings)

    # Called exactly 3 times (2 failures + 1 success)
    assert backend._client.embeddings.create.call_count == 3, (
        f"Expected 3 API calls (2 retries + success), "
        f"got {backend._client.embeddings.create.call_count}"
    )
    assert result.shape == (3, 1536), f"Expected (3, 1536), got {result.shape}"
    assert result.dtype == np.float32


def test_retry_exhaustion_raises(backend: OpenAIBackend) -> None:
    """Raises ``tenacity.RetryError`` when all retry attempts are exhausted (SC3)."""
    from tenacity import (
        RetryError,
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_none,
    )

    # Always raise RateLimitError
    backend._client.embeddings.create.side_effect = make_rate_limit_error()

    fast_retry = retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        stop=stop_after_attempt(3),
        wait=wait_none(),
    )
    backend._call_api = fast_retry(backend._raw_call)

    with pytest.raises(RetryError):
        backend.embed(["test"])


def test_non_rate_limit_error_not_retried(backend: OpenAIBackend) -> None:
    """``AuthenticationError`` propagates immediately without retry (SC3)."""
    auth_error = make_authentication_error()
    backend._client.embeddings.create.side_effect = auth_error

    with pytest.raises(openai.AuthenticationError):
        backend.embed(["test"])

    # Called exactly once -- no retry on non-rate-limit errors
    assert backend._client.embeddings.create.call_count == 1, (
        f"Expected 1 API call (no retry), "
        f"got {backend._client.embeddings.create.call_count}"
    )


# ---------------------------------------------------------------------------
# SC4 -- ImportError with install hint (test 10)
# ---------------------------------------------------------------------------


def test_import_error_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """``ImportError`` raised with install hint when openai is not installed (SC4).

    Monkeypatches ``builtins.__import__`` to block the "openai" import even
    though openai IS installed in this environment (importorskip passed).
    """
    real_import = builtins.__import__

    def mock_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "openai":
            raise ImportError("mocked: openai not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    # Reload module so the lazy import runs fresh inside __init__
    import json_semantic_diff.backends.openai as mod

    importlib.reload(mod)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")
    with pytest.raises(ImportError, match=r"pip install json-semantic-diff\[openai\]"):
        mod.OpenAIBackend()


# ---------------------------------------------------------------------------
# Additional coverage (tests 11-16)
# ---------------------------------------------------------------------------


def test_embed_returns_correct_shape(backend: OpenAIBackend) -> None:
    """``embed(3 strings)`` returns shape ``(3, 1536)`` float32."""
    strings = ["user_name", "address", "email"]
    backend._client.embeddings.create.return_value = make_mock_response(strings)

    result = backend.embed(strings)

    assert result.shape == (3, 1536), f"Expected (3, 1536), got {result.shape}"
    assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"


def test_embed_single_string(backend: OpenAIBackend) -> None:
    """``embed(1 string)`` returns shape ``(1, 1536)``."""
    strings = ["user_name"]
    backend._client.embeddings.create.return_value = make_mock_response(strings)

    result = backend.embed(strings)

    assert result.shape == (1, 1536), f"Expected (1, 1536), got {result.shape}"


def test_embed_empty_list(backend: OpenAIBackend) -> None:
    """``embed([])`` returns shape ``(0, 1536)`` float32 without API call."""
    result = backend.embed([])

    assert result.shape == (0, 1536), f"Expected (0, 1536), got {result.shape}"
    assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"
    backend._client.embeddings.create.assert_not_called()


def test_embed_preserves_input_order(backend: OpenAIBackend) -> None:
    """Output rows match input order even when API returns items in reverse (defensive test).

    The production code sorts by ``item.index`` before stacking into an ndarray.
    This test verifies that defensive sort produces correct results when the
    API returns items out of order.
    """
    strings = ["alpha", "beta", "gamma"]

    # Build response with items in reverse order
    mock_response = MagicMock()
    items = []
    for i, _ in enumerate(strings):
        item = MagicMock()
        # Use distinct embeddings to verify per-row correctness:
        # alpha -> [0.1]*1536, beta -> [0.2]*1536, gamma -> [0.3]*1536
        item.embedding = [0.1 * (i + 1)] * 1536
        item.index = i  # index 0=alpha, 1=beta, 2=gamma
        items.append(item)

    # Return items in reverse index order (gamma, beta, alpha)
    mock_response.data = list(reversed(items))
    backend._client.embeddings.create.return_value = mock_response

    result = backend.embed(strings)

    # Row 0 should be alpha ([0.1, ...]), row 2 should be gamma ([0.3, ...])
    assert result.shape == (3, 1536)
    assert np.allclose(result[0], [0.1] * 1536), "Row 0 should be alpha"
    assert np.allclose(result[1], [0.2] * 1536), "Row 1 should be beta"
    assert np.allclose(result[2], [0.3] * 1536), "Row 2 should be gamma"


def test_similarity_method_present() -> None:
    """``OpenAIBackend`` exposes ``similarity()`` per the EmbeddingBackend Protocol.

    The method delegates to ``embed()`` + clamped cosine; callers wrap in
    ``EmbeddingCache`` to avoid billing the same string twice.
    """
    assert hasattr(OpenAIBackend, "similarity")


def test_client_max_retries_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """``OpenAIBackend.__init__`` creates the OpenAI client with ``max_retries=0``.

    This prevents double-retry: tenacity (6 attempts) x SDK retry (2) = 18
    HTTP calls per ``embed()``.  tenacity is the sole retry controller.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")

    with patch("openai.OpenAI") as mock_openai_class:
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        import json_semantic_diff.backends.openai as mod

        importlib.reload(mod)
        _b = mod.OpenAIBackend()

    # Verify the OpenAI client was constructed with max_retries=0
    mock_openai_class.assert_called_once_with(max_retries=0)
