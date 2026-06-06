"""Tests for word-based chunking size and overlap arithmetic."""

from __future__ import annotations

from itertools import pairwise

import pytest

from rag_pipeline.chunking import chunk_text


def _words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_empty_input_yields_no_chunks() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\t  ") == []


def test_single_chunk_when_text_shorter_than_size() -> None:
    chunks = chunk_text(_words(5), chunk_size=10, overlap=2)
    assert len(chunks) == 1
    assert chunks[0].start_word == 0
    assert chunks[0].end_word == 5
    assert chunks[0].text == _words(5)


def test_chunk_count_and_overlap_math() -> None:
    # 100 words, size 10, overlap 2 -> step 8 -> ceil((100-10)/8)+1 = 13 chunks.
    chunks = chunk_text(_words(100), chunk_size=10, overlap=2)
    assert len(chunks) == 13
    # Every non-final chunk holds exactly chunk_size words.
    for chunk in chunks[:-1]:
        assert chunk.end_word - chunk.start_word == 10
    # Consecutive chunks advance by step = size - overlap = 8.
    for prev, nxt in pairwise(chunks):
        assert nxt.start_word - prev.start_word == 8
    # The last chunk reaches the end and is not over-sized.
    assert chunks[-1].end_word == 100


def test_overlap_words_are_shared_between_consecutive_chunks() -> None:
    chunks = chunk_text(_words(30), chunk_size=10, overlap=3)
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    # The last 3 words of chunk 0 equal the first 3 words of chunk 1.
    assert first_words[-3:] == second_words[:3]


def test_zero_overlap_is_contiguous_partition() -> None:
    chunks = chunk_text(_words(20), chunk_size=5, overlap=0)
    assert len(chunks) == 4
    rejoined = " ".join(c.text for c in chunks)
    assert rejoined == _words(20)


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (-1, 0), (5, -1), (5, 5), (5, 6)],
)
def test_invalid_parameters_raise(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text(_words(10), chunk_size=chunk_size, overlap=overlap)
