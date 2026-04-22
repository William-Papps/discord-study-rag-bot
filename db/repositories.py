from __future__ import annotations

from datetime import datetime, timezone

from db.connection import Database
from db.models import (
    ChunkRecord,
    ChunkState,
    MessageUpsertReport,
    NoteSource,
    SourceReference,
    StoredMessage,
    SyncState,
)
from utils.text import shorten


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MessageRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def upsert_many(self, messages: list[StoredMessage]) -> MessageUpsertReport:
        if not messages:
            return MessageUpsertReport(
                inserted=0,
                updated=0,
                unchanged=0,
                changed_message_ids=[],
            )
        now = utc_now()
        inserted = 0
        updated = 0
        unchanged = 0
        changed_message_ids: list[int] = []
        with self.database.transaction() as conn:
            for message in messages:
                existing = conn.execute(
                    """
                    SELECT
                        channel_id, channel_name, category_name, author_id, author_name,
                        timestamp, message_url, raw_content, cleaned_content
                    FROM messages
                    WHERE message_id = ?
                    """,
                    (message.message_id,),
                ).fetchone()
                if existing is None:
                    inserted += 1
                    changed_message_ids.append(message.message_id)
                    conn.execute(
                        """
                        INSERT INTO messages (
                            message_id, channel_id, channel_name, category_name, author_id, author_name,
                            timestamp, message_url, raw_content, cleaned_content,
                            created_at, updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            message.message_id,
                            message.channel_id,
                            message.channel_name,
                            message.category_name,
                            message.author_id,
                            message.author_name,
                            message.timestamp,
                            message.message_url,
                            message.raw_content,
                            message.cleaned_content,
                            now,
                            now,
                        ),
                    )
                    continue

                if not _stored_message_changed(existing, message):
                    unchanged += 1
                    continue

                updated += 1
                changed_message_ids.append(message.message_id)
                conn.execute(
                    """
                    UPDATE messages
                    SET
                        channel_id = ?,
                        channel_name = ?,
                        category_name = ?,
                        author_id = ?,
                        author_name = ?,
                        timestamp = ?,
                        message_url = ?,
                        raw_content = ?,
                        cleaned_content = ?,
                        updated_at = ?
                    WHERE message_id = ?
                    """,
                    (
                        message.channel_id,
                        message.channel_name,
                        message.category_name,
                        message.author_id,
                        message.author_name,
                        message.timestamp,
                        message.message_url,
                        message.raw_content,
                        message.cleaned_content,
                        now,
                        message.message_id,
                    ),
                )
        return MessageUpsertReport(
            inserted=inserted,
            updated=updated,
            unchanged=unchanged,
            changed_message_ids=changed_message_ids,
        )

    def list_for_channel(
        self,
        channel_id: int,
        after_message_id: int | None = None,
    ) -> list[StoredMessage]:
        params: tuple[object, ...]
        if after_message_id is None:
            sql = """
                SELECT * FROM messages
                WHERE channel_id = ?
                ORDER BY message_id ASC
            """
            params = (channel_id,)
        else:
            sql = """
                SELECT * FROM messages
                WHERE channel_id = ? AND message_id > ?
                ORDER BY message_id ASC
            """
            params = (channel_id, after_message_id)
        rows = self.database.fetch_all(sql, params)
        return [_message_from_row(row) for row in rows]

    def search(
        self,
        query: str,
        channel_id: int | None = None,
        limit: int = 10,
    ) -> list[StoredMessage]:
        terms = _search_terms(query)
        if not terms:
            return []

        filters = ["cleaned_content LIKE ?"]
        params: list[object] = [f"%{terms[0]}%"]
        for term in terms[1:]:
            filters.append("cleaned_content LIKE ?")
            params.append(f"%{term}%")

        channel_filter = ""
        if channel_id is not None:
            channel_filter = "AND channel_id = ?"
            params.append(channel_id)

        params.append(limit)
        rows = self.database.fetch_all(
            f"""
            SELECT * FROM messages
            WHERE {' AND '.join(filters)}
              {channel_filter}
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            tuple(params),
        )
        return [_message_from_row(row) for row in rows]

    def search_channels(
        self,
        query: str,
        channel_ids: list[int],
        limit: int = 10,
    ) -> list[StoredMessage]:
        terms = _search_terms(query)
        if not terms or not channel_ids:
            return []

        filters = ["cleaned_content LIKE ?"]
        params: list[object] = [f"%{terms[0]}%"]
        for term in terms[1:]:
            filters.append("cleaned_content LIKE ?")
            params.append(f"%{term}%")

        placeholders = ", ".join("?" for _ in channel_ids)
        params.extend(channel_ids)
        params.append(limit)
        rows = self.database.fetch_all(
            f"""
            SELECT * FROM messages
            WHERE {' AND '.join(filters)}
              AND channel_id IN ({placeholders})
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            tuple(params),
        )
        return [_message_from_row(row) for row in rows]

    def delete_many(self, message_ids: list[int]) -> int:
        if not message_ids:
            return 0
        placeholders = ", ".join("?" for _ in message_ids)
        with self.database.transaction() as conn:
            cursor = conn.execute(
                f"DELETE FROM messages WHERE message_id IN ({placeholders})",
                tuple(message_ids),
            )
            return int(cursor.rowcount)

    def count(self) -> int:
        row = self.database.fetch_one("SELECT COUNT(*) AS count FROM messages")
        return int(row["count"]) if row else 0

    def list_channel_ids(self) -> list[int]:
        rows = self.database.fetch_all(
            "SELECT DISTINCT channel_id FROM messages ORDER BY channel_id ASC"
        )
        return [int(row["channel_id"]) for row in rows]

    def list_sources(self) -> list[NoteSource]:
        rows = self.database.fetch_all(
            """
            SELECT
                m.channel_id,
                m.channel_name,
                m.category_name,
                COUNT(DISTINCT m.message_id) AS message_count,
                COUNT(DISTINCT c.chunk_id) AS chunk_count,
                MIN(m.timestamp) AS first_timestamp,
                MAX(m.timestamp) AS last_timestamp
            FROM messages m
            LEFT JOIN chunks c ON c.channel_id = m.channel_id
            GROUP BY m.channel_id, m.channel_name, m.category_name
            ORDER BY m.category_name ASC, m.channel_name ASC
            """
        )
        return [
            NoteSource(
                channel_id=int(row["channel_id"]),
                channel_name=str(row["channel_name"]),
                category_name=row["category_name"],
                message_count=int(row["message_count"]),
                chunk_count=int(row["chunk_count"]),
                first_timestamp=row["first_timestamp"],
                last_timestamp=row["last_timestamp"],
            )
            for row in rows
        ]


class ChunkRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def upsert_chunk(self, chunk: ChunkRecord) -> None:
        now = utc_now()
        with self.database.transaction() as conn:
            conn.execute(
                """
                INSERT INTO chunks (
                    chunk_id, chunk_text, channel_id, channel_name, first_timestamp,
                    category_name, last_timestamp, embedded, vector_id, content_hash, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    chunk_text = excluded.chunk_text,
                    channel_id = excluded.channel_id,
                    channel_name = excluded.channel_name,
                    category_name = excluded.category_name,
                    first_timestamp = excluded.first_timestamp,
                    last_timestamp = excluded.last_timestamp,
                    content_hash = excluded.content_hash,
                    updated_at = excluded.updated_at
                """,
                (
                    chunk.chunk_id,
                    chunk.chunk_text,
                    chunk.channel_id,
                    chunk.channel_name,
                    chunk.first_timestamp,
                    chunk.category_name,
                    chunk.last_timestamp,
                    int(chunk.embedded),
                    chunk.vector_id,
                    chunk.content_hash,
                    now,
                    now,
                ),
            )
            conn.execute("DELETE FROM chunk_messages WHERE chunk_id = ?", (chunk.chunk_id,))
            for position, message_id in enumerate(chunk.source_message_ids):
                conn.execute(
                    """
                    INSERT OR IGNORE INTO chunk_messages (chunk_id, message_id, position)
                    VALUES (?, ?, ?)
                    """,
                    (chunk.chunk_id, message_id, position),
                )

    def list_unembedded(self, limit: int) -> list[ChunkRecord]:
        rows = self.database.fetch_all(
            """
            SELECT * FROM chunks
            WHERE embedded = 0
            ORDER BY first_timestamp ASC
            LIMIT ?
            """,
            (limit,),
        )
        return [self._hydrate_chunk(row) for row in rows]

    def list_for_channel(
        self,
        channel_id: int,
        limit: int | None = None,
    ) -> list[ChunkRecord]:
        if limit is None:
            rows = self.database.fetch_all(
                """
                SELECT * FROM chunks
                WHERE channel_id = ?
                ORDER BY first_timestamp ASC
                """,
                (channel_id,),
            )
        else:
            rows = self.database.fetch_all(
                """
                SELECT * FROM chunks
                WHERE channel_id = ?
                ORDER BY first_timestamp ASC
                LIMIT ?
                """,
                (channel_id, limit),
            )
        return [self._hydrate_chunk(row) for row in rows]

    def list_for_channels(
        self,
        channel_ids: list[int],
        limit: int | None = None,
    ) -> list[ChunkRecord]:
        if not channel_ids:
            return []
        placeholders = ", ".join("?" for _ in channel_ids)
        params: list[object] = list(channel_ids)
        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT ?"
            params.append(limit)
        rows = self.database.fetch_all(
            f"""
            SELECT * FROM chunks
            WHERE channel_id IN ({placeholders})
            ORDER BY first_timestamp ASC
            {limit_clause}
            """,
            tuple(params),
        )
        return [self._hydrate_chunk(row) for row in rows]

    def search(
        self,
        query: str,
        channel_id: int | None = None,
        limit: int = 10,
    ) -> list[ChunkRecord]:
        terms = _search_terms(query)
        if not terms:
            return []

        filters = ["chunk_text LIKE ?"]
        params: list[object] = [f"%{terms[0]}%"]
        for term in terms[1:]:
            filters.append("chunk_text LIKE ?")
            params.append(f"%{term}%")

        channel_filter = ""
        if channel_id is not None:
            channel_filter = "AND channel_id = ?"
            params.append(channel_id)

        params.append(limit)
        rows = self.database.fetch_all(
            f"""
            SELECT * FROM chunks
            WHERE {' AND '.join(filters)}
              {channel_filter}
            ORDER BY first_timestamp DESC
            LIMIT ?
            """,
            tuple(params),
        )
        return [self._hydrate_chunk(row) for row in rows]

    def search_channels(
        self,
        query: str,
        channel_ids: list[int],
        limit: int = 10,
    ) -> list[ChunkRecord]:
        terms = _search_terms(query)
        if not terms or not channel_ids:
            return []

        filters = ["chunk_text LIKE ?"]
        params: list[object] = [f"%{terms[0]}%"]
        for term in terms[1:]:
            filters.append("chunk_text LIKE ?")
            params.append(f"%{term}%")

        placeholders = ", ".join("?" for _ in channel_ids)
        params.extend(channel_ids)
        params.append(limit)
        rows = self.database.fetch_all(
            f"""
            SELECT * FROM chunks
            WHERE {' AND '.join(filters)}
              AND channel_id IN ({placeholders})
            ORDER BY first_timestamp DESC
            LIMIT ?
            """,
            tuple(params),
        )
        return [self._hydrate_chunk(row) for row in rows]

    def mark_embedded(self, chunk_id: str, vector_id: str) -> None:
        self.database.execute(
            """
            UPDATE chunks
            SET embedded = 1, vector_id = ?, updated_at = ?
            WHERE chunk_id = ?
            """,
            (vector_id, utc_now(), chunk_id),
        )

    def mark_unembedded(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        placeholders = ", ".join("?" for _ in chunk_ids)
        self.database.execute(
            f"""
            UPDATE chunks
            SET embedded = 0, vector_id = NULL, updated_at = ?
            WHERE chunk_id IN ({placeholders})
            """,
            (utc_now(), *chunk_ids),
        )

    def list_embedded_chunk_ids(self) -> list[str]:
        rows = self.database.fetch_all(
            """
            SELECT chunk_id FROM chunks
            WHERE embedded = 1
            ORDER BY first_timestamp ASC
            """
        )
        return [str(row["chunk_id"]) for row in rows]

    def list_chunk_ids_for_channels(self, channel_ids: list[int]) -> list[str]:
        if not channel_ids:
            return []
        placeholders = ", ".join("?" for _ in channel_ids)
        rows = self.database.fetch_all(
            f"""
            SELECT chunk_id FROM chunks
            WHERE channel_id IN ({placeholders})
            ORDER BY first_timestamp ASC
            """,
            tuple(channel_ids),
        )
        return [str(row["chunk_id"]) for row in rows]

    def reset_chunks_for_channels(self, channel_ids: list[int]) -> None:
        if not channel_ids:
            return
        placeholders = ", ".join("?" for _ in channel_ids)
        with self.database.transaction() as conn:
            conn.execute(
                f"DELETE FROM chunks WHERE channel_id IN ({placeholders})",
                tuple(channel_ids),
            )
            conn.execute(
                f"DELETE FROM chunk_state WHERE channel_id IN ({placeholders})",
                tuple(channel_ids),
            )

    def get_source_references(self, chunk_ids: list[str]) -> list[SourceReference]:
        if not chunk_ids:
            return []
        placeholders = ", ".join("?" for _ in chunk_ids)
        rows = self.database.fetch_all(
            f"""
            SELECT
                c.chunk_id,
                c.channel_id,
                c.channel_name,
                c.category_name,
                c.chunk_text,
                m.timestamp,
                m.message_url
            FROM chunks c
            JOIN chunk_messages cm ON cm.chunk_id = c.chunk_id
            JOIN messages m ON m.message_id = cm.message_id
            WHERE c.chunk_id IN ({placeholders})
              AND cm.position = (
                  SELECT MIN(position)
                  FROM chunk_messages
                  WHERE chunk_id = c.chunk_id
              )
            ORDER BY c.first_timestamp ASC
            """,
            tuple(chunk_ids),
        )
        references = [
            SourceReference(
                chunk_id=str(row["chunk_id"]),
                channel_id=int(row["channel_id"]),
                channel_name=str(row["channel_name"]),
                category_name=row["category_name"],
                timestamp=str(row["timestamp"]),
                message_url=str(row["message_url"]),
                snippet=shorten(str(row["chunk_text"])),
            )
            for row in rows
        ]
        order = {chunk_id: index for index, chunk_id in enumerate(chunk_ids)}
        return sorted(references, key=lambda reference: order.get(reference.chunk_id, len(order)))

    def get_chunk_state(self, channel_id: int) -> ChunkState | None:
        row = self.database.fetch_one(
            "SELECT * FROM chunk_state WHERE channel_id = ?",
            (channel_id,),
        )
        if row is None:
            return None
        return ChunkState(
            channel_id=int(row["channel_id"]),
            last_chunked_message_id=row["last_chunked_message_id"],
            last_chunked_timestamp=row["last_chunked_timestamp"],
            updated_at=row["updated_at"],
        )

    def update_chunk_state(
        self,
        channel_id: int,
        last_message_id: int,
        last_timestamp: str,
    ) -> None:
        now = utc_now()
        self.database.execute(
            """
            INSERT INTO chunk_state (
                channel_id, last_chunked_message_id, last_chunked_timestamp, updated_at
            )
            VALUES (?, ?, ?, ?)
            ON CONFLICT(channel_id) DO UPDATE SET
                last_chunked_message_id = excluded.last_chunked_message_id,
                last_chunked_timestamp = excluded.last_chunked_timestamp,
                updated_at = excluded.updated_at
            """,
            (channel_id, last_message_id, last_timestamp, now),
        )

    def reset_all_chunks(self) -> None:
        with self.database.transaction() as conn:
            conn.execute("DELETE FROM chunk_messages")
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM chunk_state")

    def count(self) -> int:
        row = self.database.fetch_one("SELECT COUNT(*) AS count FROM chunks")
        return int(row["count"]) if row else 0

    def count_embedded(self) -> int:
        row = self.database.fetch_one("SELECT COUNT(*) AS count FROM chunks WHERE embedded = 1")
        return int(row["count"]) if row else 0

    def _hydrate_chunk(self, row: object) -> ChunkRecord:
        message_rows = self.database.fetch_all(
            """
            SELECT message_id FROM chunk_messages
            WHERE chunk_id = ?
            ORDER BY position ASC
            """,
            (row["chunk_id"],),
        )
        return ChunkRecord(
            chunk_id=str(row["chunk_id"]),
            source_message_ids=[int(message_row["message_id"]) for message_row in message_rows],
            chunk_text=str(row["chunk_text"]),
            channel_id=int(row["channel_id"]),
            channel_name=str(row["channel_name"]),
            category_name=row["category_name"],
            first_timestamp=str(row["first_timestamp"]),
            last_timestamp=str(row["last_timestamp"]),
            embedded=bool(row["embedded"]),
            vector_id=row["vector_id"],
            content_hash=str(row["content_hash"]),
        )


class SyncStateRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def get(self, channel_id: int) -> SyncState | None:
        row = self.database.fetch_one(
            "SELECT * FROM sync_state WHERE channel_id = ?",
            (channel_id,),
        )
        if row is None:
            return None
        return _sync_state_from_row(row)

    def upsert(
        self,
        channel_id: int,
        channel_name: str,
        last_synced_message_id: int | None,
        last_synced_timestamp: str | None,
        last_full_sync_at: str | None = None,
    ) -> None:
        existing = self.get(channel_id)
        full_sync_at = last_full_sync_at if last_full_sync_at is not None else (
            existing.last_full_sync_at if existing else None
        )
        self.database.execute(
            """
            INSERT INTO sync_state (
                channel_id, channel_name, last_synced_message_id,
                last_synced_timestamp, last_full_sync_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(channel_id) DO UPDATE SET
                channel_name = excluded.channel_name,
                last_synced_message_id = excluded.last_synced_message_id,
                last_synced_timestamp = excluded.last_synced_timestamp,
                last_full_sync_at = excluded.last_full_sync_at,
                updated_at = excluded.updated_at
            """,
            (
                channel_id,
                channel_name,
                last_synced_message_id,
                last_synced_timestamp,
                full_sync_at,
                utc_now(),
            ),
        )

    def list_all(self) -> list[SyncState]:
        rows = self.database.fetch_all(
            "SELECT * FROM sync_state ORDER BY channel_name ASC, channel_id ASC"
        )
        return [_sync_state_from_row(row) for row in rows]


def _message_from_row(row: object) -> StoredMessage:
    return StoredMessage(
        message_id=int(row["message_id"]),
        channel_id=int(row["channel_id"]),
        channel_name=str(row["channel_name"]),
        category_name=row["category_name"],
        author_id=int(row["author_id"]),
        author_name=str(row["author_name"]),
        timestamp=str(row["timestamp"]),
        message_url=str(row["message_url"]),
        raw_content=str(row["raw_content"]),
        cleaned_content=str(row["cleaned_content"]),
    )


def _stored_message_changed(row: object, message: StoredMessage) -> bool:
    return any(
        (
            int(row["channel_id"]) != message.channel_id,
            str(row["channel_name"]) != message.channel_name,
            row["category_name"] != message.category_name,
            int(row["author_id"]) != message.author_id,
            str(row["author_name"]) != message.author_name,
            str(row["timestamp"]) != message.timestamp,
            str(row["message_url"]) != message.message_url,
            str(row["raw_content"]) != message.raw_content,
            str(row["cleaned_content"]) != message.cleaned_content,
        )
    )


def _search_terms(query: str) -> list[str]:
    return [term for term in query.strip().split() if len(term) >= 2][:6]


def _sync_state_from_row(row: object) -> SyncState:
    return SyncState(
        channel_id=int(row["channel_id"]),
        channel_name=str(row["channel_name"]),
        last_synced_message_id=row["last_synced_message_id"],
        last_synced_timestamp=row["last_synced_timestamp"],
        last_full_sync_at=row["last_full_sync_at"],
        updated_at=row["updated_at"],
    )
