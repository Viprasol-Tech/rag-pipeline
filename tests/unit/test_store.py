"""Tests for cosine similarity and the in-memory vector store ranking."""

from __future__ import annotations

import math

import pytest

from rag_pipeline.embed import FakeEmbedder
from rag_pipeline.store import InMemoryVectorStore, cosine_similarity


def test_cosine_identical_and_orthogonal() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_zero_vector_is_zero() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_cosine_known_value() -> None:
    assert cosine_similarity([1.0, 1.0], [1.0, 0.0]) == pytest.approx(1.0 / math.sqrt(2))


def test_cosine_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 0.0], [1.0])


def test_search_ranks_semantically_closest_doc_first() -> None:
    embed = FakeEmbedder(dim=512)
    store = InMemoryVectorStore()
    store.add("the cat sat on the mat", embed("the cat sat on the mat"))
    store.add("stock markets rose sharply today", embed("stock markets rose sharply today"))
    store.add("a kitten napped on a rug", embed("a kitten napped on a rug"))

    hits = store.search(embed("where did the cat sleep"), k=3)
    assert len(hits) == 3
    # The cat document shares the most vocabulary with the query.
    assert hits[0].text == "the cat sat on the mat"
    # Scores are sorted descending.
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_add_returns_index_and_len_tracks() -> None:
    store = InMemoryVectorStore()
    assert store.add("a", [1.0, 0.0]) == 0
    assert store.add("b", [0.0, 1.0]) == 1
    assert len(store) == 2


def test_dim_mismatch_on_add_raises() -> None:
    store = InMemoryVectorStore()
    store.add("a", [1.0, 0.0])
    with pytest.raises(ValueError):
        store.add("b", [1.0, 0.0, 0.0])


def test_search_invalid_k_raises() -> None:
    store = InMemoryVectorStore()
    store.add("a", [1.0, 0.0])
    with pytest.raises(ValueError):
        store.search([1.0, 0.0], k=0)
