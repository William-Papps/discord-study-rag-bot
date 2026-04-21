from __future__ import annotations

import logging
from dataclasses import dataclass

from db.connection import Database
from db.repositories import ChunkRepository, MessageRepository, SyncStateRepository
from db.schema import initialize_schema
from rag.answer_generator import create_answer_generator
from rag.chunking import ChunkingConfig, DiscordChunker
from rag.embeddings import create_embedding_provider
from rag.retriever import RAGPipeline, Retriever
from rag.vector_store import ChromaVectorStore
from services.attachment_text_service import AttachmentTextService, create_attachment_text_service
from services.indexing_service import IndexingService
from services.study_service import StudyService
from services.sync_service import SyncService
from utils.config import Settings, load_settings
from utils.logging import configure_logging

logger = logging.getLogger(__name__)


@dataclass
class AppContext:
    settings: Settings
    database: Database
    message_repository: MessageRepository
    chunk_repository: ChunkRepository
    sync_state_repository: SyncStateRepository
    vector_store: ChromaVectorStore
    attachment_text_service: AttachmentTextService
    indexing_service: IndexingService
    study_service: StudyService
    sync_service: SyncService
    rag_pipeline: RAGPipeline

    def close(self) -> None:
        self.database.close()


def initialize_app() -> AppContext:
    settings = load_settings()
    configure_logging(settings.log_level)
    settings.ensure_directories()

    logger.info("Initializing SQLite database at %s", settings.database_path)
    database = Database(settings.database_path)
    initialize_schema(database)

    message_repository = MessageRepository(database)
    chunk_repository = ChunkRepository(database)
    sync_state_repository = SyncStateRepository(database)

    logger.info("Loading Chroma vector store at %s", settings.chroma_path)
    vector_store = ChromaVectorStore(
        persist_path=settings.chroma_path,
        collection_name=settings.chroma_collection,
    )

    embedding_provider = create_embedding_provider(settings)
    answer_generator = create_answer_generator(settings)
    attachment_text_service = create_attachment_text_service(settings)
    chunker = DiscordChunker(
        ChunkingConfig(
            target_tokens=settings.chunk_target_tokens,
            max_tokens=settings.chunk_max_tokens,
            overlap_messages=settings.chunk_overlap_messages,
            max_time_gap_minutes=settings.chunk_max_time_gap_minutes,
        )
    )

    indexing_service = IndexingService(
        message_repository=message_repository,
        chunk_repository=chunk_repository,
        chunker=chunker,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )
    study_service = StudyService(
        message_repository=message_repository,
        chunk_repository=chunk_repository,
        answer_generator=answer_generator,
    )
    retriever = Retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        top_k=settings.retrieval_top_k,
        min_similarity=settings.retrieval_min_similarity,
    )
    rag_pipeline = RAGPipeline(
        retriever=retriever,
        answer_generator=answer_generator,
        chunk_repository=chunk_repository,
        min_results=settings.retrieval_min_results,
    )
    sync_service = SyncService(
        settings=settings,
        message_repository=message_repository,
        chunk_repository=chunk_repository,
        sync_state_repository=sync_state_repository,
        indexing_service=indexing_service,
        vector_store=vector_store,
        attachment_text_service=attachment_text_service,
    )

    logger.info("Startup complete. Existing vectors: %s", vector_store.count())
    return AppContext(
        settings=settings,
        database=database,
        message_repository=message_repository,
        chunk_repository=chunk_repository,
        sync_state_repository=sync_state_repository,
        vector_store=vector_store,
        attachment_text_service=attachment_text_service,
        indexing_service=indexing_service,
        study_service=study_service,
        sync_service=sync_service,
        rag_pipeline=rag_pipeline,
    )
