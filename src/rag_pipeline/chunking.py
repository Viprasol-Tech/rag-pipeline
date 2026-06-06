"""Word-based text chunking with configurable size and overlap.

Splitting long documents into overlapping windows of words is the first stage
of a RAG pipeline: it bounds the size of each unit we embed and retrieve, while
the overlap preserves context that would otherwise be cut at a chunk boundary.

Part of RAG Pipeline by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A contiguous window of words taken from a source document.

    Attributes:
        text: The chunk text (words rejoined with single spaces).
        index: Zero-based position of this chunk within its document.
        start_word: Index of the first source word in this chunk.
        end_word: Index just past the last source word (exclusive).
    """

    text: str
    index: int = Field(ge=0)
    start_word: int = Field(ge=0)
    end_word: int = Field(ge=0)


def chunk_text(text: str, *, chunk_size: int = 50, overlap: int = 10) -> list[Chunk]:
    """Split ``text`` into overlapping chunks measured in words.

    The text is tokenised on whitespace. Consecutive chunks advance by
    ``chunk_size - overlap`` words, so each chunk shares exactly ``overlap``
    words with its predecessor (except the first). The final chunk is whatever
    words remain and is never duplicated.

    Args:
        text: The source text to split.
        chunk_size: Maximum number of words per chunk.
        overlap: Number of words shared between consecutive chunks.

    Returns:
        A list of :class:`Chunk` objects in document order. Empty or
        whitespace-only input yields an empty list.

    Raises:
        ValueError: If ``chunk_size`` is not positive, or ``overlap`` is
            negative or not strictly less than ``chunk_size``.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be strictly less than chunk_size")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks: list[Chunk] = []
    start = 0
    index = 0
    total = len(words)
    while start < total:
        end = min(start + chunk_size, total)
        chunks.append(
            Chunk(
                text=" ".join(words[start:end]),
                index=index,
                start_word=start,
                end_word=end,
            )
        )
        index += 1
        if end == total:
            break
        start += step
    return chunks
