"""In-memory vector store with cosine-similarity search.

A vector store keeps ``(text, embedding)`` records and answers nearest-neighbour
queries. This module implements genuine cosine similarity (no approximations)
over a plain Python list — ideal for small corpora, tests, and demos. Swap in a
real ANN backend (FAISS, pgvector, ...) behind the same ``add``/``search`` shape
for production scale.

Part of RAG Pipeline by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field


class SearchHit(BaseModel):
    """A single search result.

    Attributes:
        text: The stored text of the matched record.
        score: Cosine similarity to the query, in ``[-1.0, 1.0]``.
        index: Insertion index of the record within the store.
        metadata: Arbitrary metadata stored alongside the record.
    """

    text: str
    score: float
    index: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute the cosine similarity between two equal-length vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        The cosine similarity in ``[-1.0, 1.0]``. Returns ``0.0`` if either
        vector has zero magnitude.

    Raises:
        ValueError: If the vectors differ in length.
    """
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} != {len(b)}")
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


class InMemoryVectorStore:
    """A simple list-backed vector store supporting cosine search."""

    def __init__(self) -> None:
        """Create an empty store."""
        self._texts: list[str] = []
        self._embeddings: list[list[float]] = []
        self._metadata: list[dict[str, Any]] = []

    def __len__(self) -> int:
        """Return the number of stored records."""
        return len(self._texts)

    def add(
        self,
        text: str,
        embedding: list[float],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Add a ``(text, embedding)`` record to the store.

        Args:
            text: The text associated with the embedding.
            embedding: The embedding vector. Must match the dimensionality of
                any previously added vectors.
            metadata: Optional metadata to keep with the record.

        Returns:
            The insertion index of the new record.

        Raises:
            ValueError: If ``embedding`` is empty or its length differs from
                the existing records.
        """
        if not embedding:
            raise ValueError("embedding must be non-empty")
        if self._embeddings and len(embedding) != len(self._embeddings[0]):
            raise ValueError(
                f"embedding dim {len(embedding)} != store dim {len(self._embeddings[0])}"
            )
        index = len(self._texts)
        self._texts.append(text)
        self._embeddings.append(list(embedding))
        self._metadata.append(dict(metadata) if metadata else {})
        return index

    def search(self, query_embedding: list[float], k: int = 5) -> list[SearchHit]:
        """Return the top-``k`` records most similar to ``query_embedding``.

        Results are ordered by descending cosine similarity; ties break by
        ascending insertion index for determinism.

        Args:
            query_embedding: The query vector.
            k: Maximum number of hits to return.

        Returns:
            Up to ``k`` :class:`SearchHit` objects. Empty if the store is empty.

        Raises:
            ValueError: If ``k`` is not positive.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        scored = [
            SearchHit(
                text=self._texts[i],
                score=cosine_similarity(query_embedding, emb),
                index=i,
                metadata=dict(self._metadata[i]),
            )
            for i, emb in enumerate(self._embeddings)
        ]
        scored.sort(key=lambda hit: (-hit.score, hit.index))
        return scored[:k]
