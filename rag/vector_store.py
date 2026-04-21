from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from db.models import ChunkRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VectorSearchResult:
    chunk_id: str
    document: str
    metadata: dict[str, Any]
    distance: float
    similarity: float


class ChromaVectorStore:
    def __init__(self, persist_path: Path, collection_name: str) -> None:
        self.persist_path = persist_path
        self.collection_name = collection_name
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_path))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk and embedding counts do not match.")

        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.chunk_text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "channel_id": chunk.channel_id,
                    "channel_name": chunk.channel_name,
                    "category_name": chunk.category_name or "",
                    "first_timestamp": chunk.first_timestamp,
                    "last_timestamp": chunk.last_timestamp,
                    "source_message_ids": ",".join(str(value) for value in chunk.source_message_ids),
                    "content_hash": chunk.content_hash,
                }
                for chunk in chunks
            ],
        )

    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        channel_id: int | None = None,
    ) -> list[VectorSearchResult]:
        where = {"channel_id": channel_id} if channel_id is not None else None
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        output: list[VectorSearchResult] = []
        for index, chunk_id in enumerate(ids):
            distance = float(distances[index])
            output.append(
                VectorSearchResult(
                    chunk_id=str(chunk_id),
                    document=str(documents[index]),
                    metadata=dict(metadatas[index] or {}),
                    distance=distance,
                    similarity=max(0.0, 1.0 - distance),
                )
            )
        return output

    def existing_ids(self, chunk_ids: list[str]) -> set[str]:
        if not chunk_ids:
            return set()
        results = self.collection.get(ids=chunk_ids)
        return {str(chunk_id) for chunk_id in results.get("ids", [])}

    def delete(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        self.collection.delete(ids=chunk_ids)

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception as exc:
            if "does not exist" not in str(exc).lower():
                raise
            logger.debug("Ignoring missing Chroma collection during reset: %s", exc)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return int(self.collection.count())
