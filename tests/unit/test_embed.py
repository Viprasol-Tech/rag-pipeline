"""Tests for the deterministic offline FakeEmbedder."""

from __future__ import annotations

import pytest

from rag_pipeline.embed import Embedder, FakeEmbedder


def test_embedder_is_deterministic() -> None:
    embed = FakeEmbedder(dim=64)
    assert embed("hello world") == embed("hello world")


def test_dimension_is_respected() -> None:
    embed = FakeEmbedder(dim=128)
    vec = embed("some text here")
    assert len(vec) == 128
    assert embed.dim == 128


def test_token_free_text_is_all_zero() -> None:
    embed = FakeEmbedder(dim=32)
    assert embed("   ... !!!   ") == [0.0] * 32


def test_case_and_punctuation_insensitive() -> None:
    embed = FakeEmbedder(dim=64)
    assert embed("Hello, World!") == embed("hello world")


def test_satisfies_embedder_protocol() -> None:
    embed = FakeEmbedder(dim=16)
    assert isinstance(embed, Embedder)


def test_invalid_dim_raises() -> None:
    with pytest.raises(ValueError):
        FakeEmbedder(dim=0)
