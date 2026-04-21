from __future__ import annotations

import logging
from dataclasses import dataclass

from db.repositories import ChunkRepository, MessageRepository
from rag.chunking import DiscordChunker
from rag.embeddings import EmbeddingProvider
from rag.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexingReport:
    chunks_created: int
    chunks_embedded: int
    channels_reindexed: int = 0
    stale_vectors_repaired: int = 0


class IndexingService:
    def __init__(
        self,
        message_repository: MessageRepository,
        chunk_repository: ChunkRepository,
        chunker: DiscordChunker,
        embedding_provider: EmbeddingProvider,
        vector_store: ChromaVectorStore,
    ) -> None:
        self.message_repository = message_repository
        self.chunk_repository = chunk_repository
        self.chunker = chunker
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def build_missing_chunks(self, channel_ids: list[int]) -> int:
        total_created = 0
        for channel_id in channel_ids:
            state = self.chunk_repository.get_chunk_state(channel_id)
            after_message_id = state.last_chunked_message_id if state else None
            messages = self.message_repository.list_for_channel(
                channel_id=channel_id,
                after_message_id=after_message_id,
            )
            if not messages:
                continue

            chunks = self.chunker.chunk_messages(messages)
            for chunk in chunks:
                self.chunk_repository.upsert_chunk(chunk)
            last_message = messages[-1]
            self.chunk_repository.update_chunk_state(
                channel_id=channel_id,
                last_message_id=last_message.message_id,
                last_timestamp=last_message.timestamp,
            )
            total_created += len(chunks)
            logger.info(
                "Chunked %s messages into %s chunks for channel %s",
                len(messages),
                len(chunks),
                channel_id,
            )
        return total_created

    def embed_unembedded_chunks(self, batch_size: int = 64) -> int:
        total_embedded = 0
        while True:
            chunks = self.chunk_repository.list_unembedded(limit=batch_size)
            if not chunks:
                return total_embedded

            embeddings = self.embedding_provider.embed_texts(
                [chunk.chunk_text for chunk in chunks]
            )
            self.vector_store.upsert_chunks(chunks, embeddings)
            for chunk in chunks:
                self.chunk_repository.mark_embedded(chunk.chunk_id, chunk.chunk_id)
            total_embedded += len(chunks)
            logger.info("Embedded %s chunks", len(chunks))

    def repair_missing_vectors(self, batch_size: int = 500) -> int:
        chunk_ids = self.chunk_repository.list_embedded_chunk_ids()
        missing: list[str] = []
        for start in range(0, len(chunk_ids), batch_size):
            batch = chunk_ids[start : start + batch_size]
            existing = self.vector_store.existing_ids(batch)
            missing.extend(chunk_id for chunk_id in batch if chunk_id not in existing)

        if missing:
            logger.warning(
                "Found %s SQLite chunks marked embedded but missing from Chroma; "
                "marking them for re-embedding",
                len(missing),
            )
            self.chunk_repository.mark_unembedded(missing)
        return len(missing)

    def index_new_data(self, channel_ids: list[int]) -> IndexingReport:
        chunks_created = self.build_missing_chunks(channel_ids)
        repaired = self.repair_missing_vectors()
        chunks_embedded = self.embed_unembedded_chunks()
        return IndexingReport(
            chunks_created=chunks_created,
            chunks_embedded=chunks_embedded,
            stale_vectors_repaired=repaired,
        )

    def reindex_channels(self, channel_ids: list[int]) -> IndexingReport:
        unique_channel_ids = sorted(set(channel_ids))
        if not unique_channel_ids:
            return IndexingReport(chunks_created=0, chunks_embedded=0)

        chunk_ids = self.chunk_repository.list_chunk_ids_for_channels(unique_channel_ids)
        logger.info(
            "Reindexing %s changed channels and deleting %s old vectors",
            len(unique_channel_ids),
            len(chunk_ids),
        )
        self.vector_store.delete(chunk_ids)
        self.chunk_repository.reset_chunks_for_channels(unique_channel_ids)
        report = self.index_new_data(unique_channel_ids)
        return IndexingReport(
            chunks_created=report.chunks_created,
            chunks_embedded=report.chunks_embedded,
            channels_reindexed=len(unique_channel_ids),
            stale_vectors_repaired=report.stale_vectors_repaired,
        )

    def reindex_all(self, channel_ids: list[int]) -> IndexingReport:
        logger.info("Resetting local chunks and vector collection before reindex")
        self.vector_store.reset()
        self.chunk_repository.reset_all_chunks()
        report = self.index_new_data(channel_ids)
        return IndexingReport(
            chunks_created=report.chunks_created,
            chunks_embedded=report.chunks_embedded,
            channels_reindexed=len(set(channel_ids)),
            stale_vectors_repaired=report.stale_vectors_repaired,
        )

    def reindex_all_local_messages(self) -> IndexingReport:
        channel_ids = self.message_repository.list_channel_ids()
        logger.info("Reindexing all local messages across %s channels", len(channel_ids))
        return self.reindex_all(channel_ids)
