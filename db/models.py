from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StoredMessage:
    message_id: int
    channel_id: int
    channel_name: str
    category_name: str | None
    author_id: int
    author_name: str
    timestamp: str
    message_url: str
    raw_content: str
    cleaned_content: str


@dataclass(frozen=True)
class MessageUpsertReport:
    inserted: int
    updated: int
    unchanged: int
    changed_message_ids: list[int]

    @property
    def total(self) -> int:
        return self.inserted + self.updated + self.unchanged


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    source_message_ids: list[int]
    chunk_text: str
    channel_id: int
    channel_name: str
    category_name: str | None
    first_timestamp: str
    last_timestamp: str
    embedded: bool
    vector_id: str | None
    content_hash: str


@dataclass(frozen=True)
class SyncState:
    channel_id: int
    channel_name: str
    last_synced_message_id: int | None
    last_synced_timestamp: str | None
    last_full_sync_at: str | None
    updated_at: str | None


@dataclass(frozen=True)
class ChunkState:
    channel_id: int
    last_chunked_message_id: int | None
    last_chunked_timestamp: str | None
    updated_at: str | None


@dataclass(frozen=True)
class SourceReference:
    chunk_id: str
    channel_name: str
    category_name: str | None
    timestamp: str
    message_url: str
    snippet: str


@dataclass(frozen=True)
class NoteSource:
    channel_id: int
    channel_name: str
    category_name: str | None
    message_count: int
    chunk_count: int
    first_timestamp: str | None
    last_timestamp: str | None


@dataclass(frozen=True)
class NoteCategory:
    name: str
    channel_ids: list[int]
    source_count: int
    message_count: int
    chunk_count: int


@dataclass(frozen=True)
class StatusReport:
    total_messages: int
    total_chunks: int
    embedded_chunks: int
    vector_count: int
    vector_path: str
    sync_states: list[SyncState]
