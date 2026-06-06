"""Command-line interface for RAG Pipeline.

``rag-pipeline demo`` ingests a small built-in corpus and answers a question
fully offline using the built-in fake embedder and echo LLM — no API keys,
no network, no heavy SDKs.

Part of RAG Pipeline by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

import typer
from rich.console import Console

from rag_pipeline import __version__
from rag_pipeline.pipeline import RAGPipeline

app = typer.Typer(add_completion=False, help="RAG Pipeline — by Viprasol Tech.")
console = Console()

_CORPUS: dict[str, str] = {
    "photosynthesis": (
        "Photosynthesis is the process by which green plants and some other "
        "organisms use sunlight to synthesize foods from carbon dioxide and "
        "water. It produces oxygen as a byproduct and stores energy in glucose."
    ),
    "mitochondria": (
        "Mitochondria are membrane-bound organelles that generate most of the "
        "chemical energy needed to power a cell. They are often called the "
        "powerhouse of the cell and produce ATP through cellular respiration."
    ),
    "water_cycle": (
        "The water cycle describes how water evaporates from the surface, rises "
        "into the atmosphere, cools and condenses into clouds, and falls back to "
        "the surface as precipitation such as rain and snow."
    ),
}


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"rag-pipeline [bold cyan]{__version__}[/] - by Viprasol Tech")


@app.command()
def demo(
    question: str = typer.Option(
        "How do plants make food from sunlight?",
        help="Question to answer against the built-in corpus.",
    ),
    top_k: int = typer.Option(2, help="Number of chunks to retrieve."),
) -> None:
    """Ingest a small corpus and answer a question, fully offline."""
    pipeline = RAGPipeline(top_k=top_k, chunk_size=20, overlap=5)
    total = 0
    for source, text in _CORPUS.items():
        total += pipeline.ingest(text, source=source)

    console.print(f"Ingested [bold]{total}[/] chunks from {len(_CORPUS)} documents.")
    console.print(f"Question: [bold]{question}[/]\n")

    result = pipeline.answer(question)

    console.print("[bold]Retrieved context:[/]")
    for rank, hit in enumerate(result.contexts, start=1):
        src = hit.metadata.get("source", "?")
        console.print(f"  {rank}. [dim](score={hit.score:.3f}, source={src})[/] {hit.text}")

    console.print(f"\n[bold green]Answer:[/] {result.answer}")


if __name__ == "__main__":
    app()
