"""Tests for the end-to-end RAG pipeline and prompt building."""

from __future__ import annotations

from rag_pipeline.pipeline import RAGPipeline, build_prompt
from rag_pipeline.store import SearchHit


def test_build_prompt_contains_question_and_context() -> None:
    hits = [SearchHit(text="the sky is blue", score=0.9, index=0)]
    prompt = build_prompt("what colour is the sky", hits)
    assert "the sky is blue" in prompt
    assert "what colour is the sky" in prompt
    assert prompt.rstrip().endswith("Answer:")
    assert "[1] the sky is blue" in prompt


def test_build_prompt_handles_no_context() -> None:
    prompt = build_prompt("anything", [])
    assert "(no context retrieved)" in prompt


def test_pipeline_retrieves_relevant_chunk_and_builds_prompt() -> None:
    pipeline = RAGPipeline(top_k=1, chunk_size=12, overlap=3)
    pipeline.ingest(
        "Photosynthesis lets plants convert sunlight into chemical energy stored as glucose.",
        source="bio",
    )
    pipeline.ingest(
        "The stock market rallied today as technology shares climbed on strong earnings.",
        source="finance",
    )

    result = pipeline.answer("How do plants use sunlight?")

    assert len(result.contexts) == 1
    top = result.contexts[0]
    # The biology chunk must outrank the finance chunk for this question.
    assert "Photosynthesis" in top.text
    assert top.metadata.get("source") == "bio"
    # The built prompt must contain the retrieved chunk and the question.
    assert top.text in result.prompt
    assert "How do plants use sunlight?" in result.prompt
    # The default echo LLM grounds its answer in the retrieved chunk.
    assert "Photosynthesis" in result.answer


def test_pipeline_uses_pluggable_llm() -> None:
    def shouting_llm(prompt: str) -> str:
        return "ANSWERED"

    pipeline = RAGPipeline(llm=shouting_llm, top_k=1)
    pipeline.ingest("some grounding text about apples and oranges")
    result = pipeline.answer("tell me about fruit")
    assert result.answer == "ANSWERED"


def test_ingest_returns_chunk_count() -> None:
    pipeline = RAGPipeline(chunk_size=5, overlap=1)
    count = pipeline.ingest(" ".join(f"word{i}" for i in range(20)))
    assert count == len(pipeline.store)
    assert count > 1
