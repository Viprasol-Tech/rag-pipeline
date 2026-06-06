"""End-to-end RAG pipeline: chunk -> embed -> store -> retrieve -> answer.

:class:`RAGPipeline` wires the four stages together and calls a pluggable LLM
to produce a grounded answer. Both the embedder and the LLM are plain callables,
so the whole pipeline runs offline with the built-in fakes, yet a single
constructor argument swaps in a real model.

Part of RAG Pipeline by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from pydantic import BaseModel, Field

from rag_pipeline.chunking import chunk_text
from rag_pipeline.embed import Embedder, FakeEmbedder
from rag_pipeline.store import InMemoryVectorStore, SearchHit

# A language model is any callable that maps a prompt to a completion string.
LLM = Callable[[str], str]

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using only the provided "
    "context. If the context does not contain the answer, say you do not know."
)


def build_prompt(
    question: str,
    contexts: Sequence[SearchHit],
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Render a grounded RAG prompt from a question and retrieved contexts.

    Args:
        question: The natural-language question to answer.
        contexts: Retrieved hits, highest-scoring first.
        system_prompt: Instruction prepended to the prompt.

    Returns:
        A single prompt string with numbered context blocks, the question,
        and a trailing ``Answer:`` cue.
    """
    if contexts:
        blocks = [f"[{i + 1}] {hit.text}" for i, hit in enumerate(contexts)]
        context_section = "\n\n".join(blocks)
    else:
        context_section = "(no context retrieved)"
    return f"{system_prompt}\n\nContext:\n{context_section}\n\nQuestion: {question}\n\nAnswer:"


def echo_llm(prompt: str) -> str:
    """A deterministic, offline LLM that grounds its answer in the context.

    It performs no model call. Instead it extracts the question and the first
    context block from the prompt and returns a reproducible grounded answer,
    so end-to-end behaviour is fully testable without a network or API key.

    Args:
        prompt: A prompt produced by :func:`build_prompt`.

    Returns:
        A grounded answer string.
    """
    question = ""
    snippet = ""
    for line in prompt.splitlines():
        if line.startswith("Question:"):
            question = line[len("Question:") :].strip()
        elif line.startswith("[1] ") and not snippet:
            snippet = line[len("[1] ") :].strip()
    if not snippet:
        return "I don't know based on the provided context."
    lead = f"Regarding '{question}': " if question else ""
    return f"{lead}{snippet}"


class RAGResult(BaseModel):
    """The outcome of a single :meth:`RAGPipeline.answer` call.

    Attributes:
        question: The original question.
        answer: The LLM's generated answer.
        prompt: The exact prompt sent to the LLM.
        contexts: The retrieved hits used as grounding, best first.
    """

    question: str
    answer: str
    prompt: str
    contexts: list[SearchHit] = Field(default_factory=list)


class RAGPipeline:
    """A complete retrieval-augmented-generation pipeline.

    Defaults to fully offline components (:class:`FakeEmbedder`,
    :class:`InMemoryVectorStore`, and :func:`echo_llm`), so it runs with no
    network and no API keys. Pass your own ``embedder`` and ``llm`` callables
    to use real models.
    """

    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
        top_k: int = 4,
        chunk_size: int = 50,
        overlap: int = 10,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        """Create a pipeline.

        Args:
            embedder: Text-to-vector callable. Defaults to :class:`FakeEmbedder`.
            llm: Prompt-to-text callable. Defaults to :func:`echo_llm`.
            top_k: Number of chunks to retrieve per question.
            chunk_size: Words per chunk when ingesting.
            overlap: Word overlap between consecutive chunks.
            system_prompt: Instruction used when building prompts.

        Raises:
            ValueError: If ``top_k`` is not positive.
        """
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.embedder: Embedder = embedder if embedder is not None else FakeEmbedder()
        self.llm: LLM = llm if llm is not None else echo_llm
        self.store = InMemoryVectorStore()
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.system_prompt = system_prompt

    def ingest(self, text: str, *, source: str | None = None) -> int:
        """Chunk ``text``, embed each chunk, and add it to the store.

        Args:
            text: The document text to ingest.
            source: Optional source label stored as chunk metadata.

        Returns:
            The number of chunks ingested.
        """
        chunks = chunk_text(text, chunk_size=self.chunk_size, overlap=self.overlap)
        for chunk in chunks:
            metadata: dict[str, object] = {"index": chunk.index}
            if source is not None:
                metadata["source"] = source
            self.store.add(chunk.text, self.embedder(chunk.text), metadata=metadata)
        return len(chunks)

    def retrieve(self, question: str, *, top_k: int | None = None) -> list[SearchHit]:
        """Embed ``question`` and return the most relevant stored chunks.

        Args:
            question: The natural-language query.
            top_k: Optional override of the configured ``top_k``.

        Returns:
            Up to ``top_k`` :class:`SearchHit` objects, best first.
        """
        k = top_k if top_k is not None else self.top_k
        return self.store.search(self.embedder(question), k=k)

    def answer(self, question: str, *, top_k: int | None = None) -> RAGResult:
        """Retrieve context, build a grounded prompt, and call the LLM.

        Args:
            question: The question to answer.
            top_k: Optional override of the configured ``top_k``.

        Returns:
            A :class:`RAGResult` with the answer, prompt, and contexts.
        """
        contexts = self.retrieve(question, top_k=top_k)
        prompt = build_prompt(question, contexts, system_prompt=self.system_prompt)
        answer = self.llm(prompt)
        return RAGResult(question=question, answer=answer, prompt=prompt, contexts=contexts)
