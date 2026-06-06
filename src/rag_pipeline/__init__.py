"""RAG Pipeline — a provider-agnostic, offline-capable Retrieval-Augmented
Generation pipeline by Viprasol Tech.

Public surface: chunking, embedding, an in-memory vector store, and a
:class:`~rag_pipeline.pipeline.RAGPipeline` that ties them together with a
pluggable LLM callable.
"""

from __future__ import annotations

from rag_pipeline.chunking import Chunk, chunk_text
from rag_pipeline.embed import Embedder, FakeEmbedder
from rag_pipeline.pipeline import RAGPipeline, RAGResult, build_prompt
from rag_pipeline.store import InMemoryVectorStore, SearchHit, cosine_similarity

__version__ = "0.1.0"
__author__ = "Viprasol Tech Private Limited"
__all__ = [
    "Chunk",
    "Embedder",
    "FakeEmbedder",
    "InMemoryVectorStore",
    "RAGPipeline",
    "RAGResult",
    "SearchHit",
    "__version__",
    "build_prompt",
    "chunk_text",
    "cosine_similarity",
]
