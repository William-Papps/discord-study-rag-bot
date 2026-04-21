from __future__ import annotations

from db.connection import Database


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS messages (
    message_id INTEGER PRIMARY KEY,
    channel_id INTEGER NOT NULL,
    channel_name TEXT NOT NULL,
    category_name TEXT,
    author_id INTEGER NOT NULL,
    author_name TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    message_url TEXT NOT NULL,
    raw_content TEXT NOT NULL,
    cleaned_content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_channel_timestamp
ON messages (channel_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_messages_channel_id
ON messages (channel_id, message_id);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    chunk_text TEXT NOT NULL,
    channel_id INTEGER NOT NULL,
    channel_name TEXT NOT NULL,
    category_name TEXT,
    first_timestamp TEXT NOT NULL,
    last_timestamp TEXT NOT NULL,
    embedded INTEGER NOT NULL DEFAULT 0,
    vector_id TEXT,
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_channel_timestamp
ON chunks (channel_id, first_timestamp);

CREATE INDEX IF NOT EXISTS idx_chunks_embedded
ON chunks (embedded);

CREATE TABLE IF NOT EXISTS chunk_messages (
    chunk_id TEXT NOT NULL,
    message_id INTEGER NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY (chunk_id, message_id),
    FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    FOREIGN KEY (message_id) REFERENCES messages(message_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS sync_state (
    channel_id INTEGER PRIMARY KEY,
    channel_name TEXT NOT NULL,
    last_synced_message_id INTEGER,
    last_synced_timestamp TEXT,
    last_full_sync_at TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunk_state (
    channel_id INTEGER PRIMARY KEY,
    last_chunked_message_id INTEGER,
    last_chunked_timestamp TEXT,
    updated_at TEXT NOT NULL
);
"""


def initialize_schema(database: Database) -> None:
    database.executescript(SCHEMA_SQL)
    _ensure_column(database, "messages", "category_name", "TEXT")
    _ensure_column(database, "chunks", "category_name", "TEXT")


def _ensure_column(database: Database, table: str, column: str, definition: str) -> None:
    rows = database.fetch_all(f"PRAGMA table_info({table})")
    existing = {str(row["name"]) for row in rows}
    if column not in existing:
        database.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
