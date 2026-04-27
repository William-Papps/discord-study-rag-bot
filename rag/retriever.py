from __future__ import annotations

from dataclasses import dataclass

from db.models import SourceReference
from db.repositories import ChunkRepository
from rag.answer_generator import AnswerContext, AnswerGenerator, REFUSAL_MESSAGE
from rag.embeddings import EmbeddingProvider
from rag.vector_store import ChromaVectorStore
from utils.time_scope import chunk_overlaps_time_scope, parse_time_scope


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    similarity: float


@dataclass(frozen=True)
class RAGAnswer:
    answer: str
    refused: bool
    sources: list[SourceReference]
    retrieved_chunks: list[RetrievedChunk]


class Retriever:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: ChromaVectorStore,
        top_k: int,
        min_similarity: float,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.top_k = top_k
        self.min_similarity = min_similarity

    def retrieve(self, question: str, channel_id: int | None = None) -> list[RetrievedChunk]:
        query_embedding = self.embedding_provider.embed_query(question)
        time_scope = parse_time_scope(question)
        top_k = self.top_k if time_scope is None else max(self.top_k * 10, 50)
        results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            channel_id=channel_id,
        )
        return [
            RetrievedChunk(
                chunk_id=result.chunk_id,
                text=result.document,
                similarity=result.similarity,
            )
            for result in results
            if result.similarity >= self.min_similarity
            and chunk_overlaps_time_scope(
                str(result.metadata.get("first_timestamp") or ""),
                str(result.metadata.get("last_timestamp") or ""),
                time_scope,
            )
        ][: self.top_k]


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        answer_generator: AnswerGenerator,
        chunk_repository: ChunkRepository,
        min_results: int,
    ) -> None:
        self.retriever = retriever
        self.answer_generator = answer_generator
        self.chunk_repository = chunk_repository
        self.min_results = min_results

    def answer_question(self, question: str, channel_id: int | None = None) -> RAGAnswer:
        retrieved = self.retriever.retrieve(question, channel_id=channel_id)
        return self.answer_from_chunks(question, retrieved)

    def answer_from_chunks(
        self,
        question: str,
        retrieved: list[RetrievedChunk],
        require_min_results: bool = True,
    ) -> RAGAnswer:
        if not retrieved or (require_min_results and len(retrieved) < self.min_results):
            return RAGAnswer(
                answer=REFUSAL_MESSAGE,
                refused=True,
                sources=[],
                retrieved_chunks=retrieved,
            )

        contexts = [
            AnswerContext(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                similarity=chunk.similarity,
            )
            for chunk in retrieved
        ]
        generated = self.answer_generator.generate(question, contexts)
        if generated.refused:
            return RAGAnswer(
                answer=REFUSAL_MESSAGE,
                refused=True,
                sources=[],
                retrieved_chunks=retrieved,
            )

        sources = self.chunk_repository.get_source_references(
            [chunk.chunk_id for chunk in retrieved]
        )
        return RAGAnswer(
            answer=generated.answer,
            refused=False,
            sources=sources,
            retrieved_chunks=retrieved,
        )
