"""Runnable end-to-end demo of the RAG pipeline — fully offline.

Run with::

    python examples/demo.py

No API keys, no network, no model downloads: it uses the built-in deterministic
``FakeEmbedder`` and ``echo_llm``.

Part of RAG Pipeline by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from rag_pipeline import RAGPipeline

CORPUS: dict[str, str] = {
    "photosynthesis": (
        "Photosynthesis is the process by which green plants use sunlight to "
        "synthesize food from carbon dioxide and water, releasing oxygen as a "
        "byproduct and storing energy in glucose."
    ),
    "mitochondria": (
        "Mitochondria are organelles that generate most of the chemical energy "
        "needed to power a cell, producing ATP through cellular respiration."
    ),
    "water_cycle": (
        "The water cycle describes how water evaporates, condenses into clouds, "
        "and falls back to the surface as precipitation such as rain and snow."
    ),
}


def main() -> None:
    """Ingest a small corpus and answer a question, end to end."""
    pipeline = RAGPipeline(top_k=2, chunk_size=20, overlap=5)

    total = 0
    for source, text in CORPUS.items():
        total += pipeline.ingest(text, source=source)
    print(f"Ingested {total} chunks from {len(CORPUS)} documents.\n")

    question = "How do plants make food from sunlight?"
    result = pipeline.answer(question)

    print(f"Question: {question}\n")
    print("Retrieved context:")
    for rank, hit in enumerate(result.contexts, start=1):
        source = hit.metadata.get("source", "?")
        print(f"  {rank}. (score={hit.score:.3f}, source={source}) {hit.text}")

    print(f"\nAnswer: {result.answer}")


if __name__ == "__main__":
    main()
