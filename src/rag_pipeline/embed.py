"""Embedding providers.

An embedder turns a string into a fixed-length vector of floats. Real systems
back this with a model (OpenAI, sentence-transformers, ...); for fully offline
and deterministic runs this module ships :class:`FakeEmbedder`, a hashed
bag-of-words embedder that requires no network, no API key, and no heavy SDK.

The :class:`Embedder` Protocol means *any* ``Callable[[str], list[float]]`` with
the right shape is a valid provider — plug in your own model freely.

Part of RAG Pipeline by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, runtime_checkable

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Embedder(Protocol):
    """Anything that maps text to a fixed-length embedding vector.

    Implementations must return vectors of the same ``dim`` for every call so
    they can be compared with cosine similarity.
    """

    dim: int

    def __call__(self, text: str) -> list[float]:
        """Return the embedding vector for ``text``."""
        ...


def _tokenize(text: str) -> list[str]:
    """Lower-case ``text`` and split into alphanumeric word tokens."""
    return _TOKEN_RE.findall(text.lower())


class FakeEmbedder:
    """Deterministic, offline hashed bag-of-words embedder.

    Each token is hashed into one of ``dim`` buckets and contributes a signed
    weight, producing a sparse term-frequency style vector. Because the hashing
    is stable, identical text always yields an identical vector, and documents
    that share vocabulary land near each other in cosine space — enough to make
    retrieval meaningful in tests and demos without any model download.

    Attributes:
        dim: Dimensionality of the produced vectors.
    """

    def __init__(self, dim: int = 256) -> None:
        """Create a fake embedder.

        Args:
            dim: Number of dimensions in each output vector. Must be positive.

        Raises:
            ValueError: If ``dim`` is not positive.
        """
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def __call__(self, text: str) -> list[float]:
        """Embed ``text`` into a length-``dim`` vector via hashed bag-of-words.

        Args:
            text: The text to embed.

        Returns:
            A list of ``dim`` floats. All-zero only for token-free input.
        """
        vector = [0.0] * self.dim
        for token in _tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign
        return vector
