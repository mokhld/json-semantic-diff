"""EmbeddingBackend Protocol for json-semantic-diff backend extension point.

Defines the structural interface all embedding backends must satisfy.
Users can plug in custom backends without inheriting from any base class —
any class with conformant ``embed`` and ``similarity`` methods passes
``isinstance`` checks.

Example::

    import numpy as np
    from json_semantic_diff.protocols import EmbeddingBackend

    class MyBackend:
        def embed(self, strings: list[str]) -> np.ndarray:
            # Return shape (N, D) float64 embedding matrix
            return np.zeros((len(strings), 768))

        def similarity(self, a: str, b: str) -> float:
            # Return a float in [0.0, 1.0]
            return 1.0 if a == b else 0.0

    assert isinstance(MyBackend(), EmbeddingBackend)  # True — structural conformance
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Structural protocol for embedding backends.

    Any class implementing both ``embed(self, strings: list[str]) -> np.ndarray``
    and ``similarity(self, a: str, b: str) -> float`` satisfies this protocol at
    runtime — no inheritance required.

    The ``embed`` method must:
    - Accept a list of strings as input.
    - Return a 2-D numpy array of shape ``(len(strings), D)`` for some embedding
      dimension ``D >= 1``.
    - Return dtype ``float64`` (or compatible floating-point dtype).

    The ``similarity`` method must:
    - Accept two strings.
    - Return a float in ``[0.0, 1.0]`` where ``1.0`` means identical and ``0.0``
      means completely dissimilar.

    Note:
        Backends that only know how to ``embed`` can still satisfy this protocol
        by wrapping themselves in :class:`json_semantic_diff.cache.EmbeddingCache`,
        which provides a cosine-based ``similarity`` fallback.
    """

    def embed(self, strings: list[str]) -> np.ndarray: ...

    def similarity(self, a: str, b: str) -> float: ...
