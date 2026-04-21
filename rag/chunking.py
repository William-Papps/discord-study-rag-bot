from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from db.models import ChunkRecord, StoredMessage
from utils.text import content_hash, estimate_token_count, parse_iso_datetime


@dataclass(frozen=True)
class ChunkingConfig:
    target_tokens: int
    max_tokens: int
    overlap_messages: int
    max_time_gap_minutes: int


@dataclass(frozen=True)
class _ChunkMessage:
    message: StoredMessage
    rendered: str
    tokens: int


class DiscordChunker:
    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config

    def chunk_messages(self, messages: list[StoredMessage]) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        current: list[_ChunkMessage] = []
        current_tokens = 0

        for message in messages:
            if not message.cleaned_content.strip():
                continue
            rendered = self._render_message(message)
            token_count = estimate_token_count(rendered)

            if current and self._should_start_new_chunk(current, current_tokens, message, token_count):
                chunks.append(self._build_chunk(current))
                current = self._overlap_tail(current)
                current_tokens = sum(item.tokens for item in current)
                if current_tokens + token_count > self.config.max_tokens:
                    current = []
                    current_tokens = 0

            current.append(_ChunkMessage(message=message, rendered=rendered, tokens=token_count))
            current_tokens += token_count

        if current:
            chunks.append(self._build_chunk(current))

        return chunks

    def _should_start_new_chunk(
        self,
        current: list[_ChunkMessage],
        current_tokens: int,
        next_message: StoredMessage,
        next_tokens: int,
    ) -> bool:
        if current_tokens + next_tokens > self.config.max_tokens:
            return True
        if (
            current_tokens >= self.config.target_tokens
            and current_tokens + next_tokens > self.config.target_tokens
        ):
            return True

        previous_timestamp = parse_iso_datetime(current[-1].message.timestamp)
        next_timestamp = parse_iso_datetime(next_message.timestamp)
        gap = next_timestamp - previous_timestamp
        return gap > timedelta(minutes=self.config.max_time_gap_minutes)

    def _overlap_tail(self, current: list[_ChunkMessage]) -> list[_ChunkMessage]:
        if self.config.overlap_messages <= 0:
            return []
        return current[-self.config.overlap_messages :]

    def _build_chunk(self, items: list[_ChunkMessage]) -> ChunkRecord:
        first = items[0].message
        last = items[-1].message
        body = "\n\n".join(item.rendered for item in items)
        heading = (
            f"Category: {first.category_name}\nChannel: #{first.channel_name}"
            if first.category_name
            else f"Channel: #{first.channel_name}"
        )
        text = f"{heading}\n\n{body}"
        digest = content_hash(text)
        chunk_id = f"chunk_{first.channel_id}_{first.message_id}_{last.message_id}_{digest[:12]}"

        return ChunkRecord(
            chunk_id=chunk_id,
            source_message_ids=[item.message.message_id for item in items],
            chunk_text=text,
            channel_id=first.channel_id,
            channel_name=first.channel_name,
            category_name=first.category_name,
            first_timestamp=first.timestamp,
            last_timestamp=last.timestamp,
            embedded=False,
            vector_id=None,
            content_hash=digest,
        )

    @staticmethod
    def _render_message(message: StoredMessage) -> str:
        return f"[{message.timestamp}] {message.author_name}: {message.cleaned_content}"
