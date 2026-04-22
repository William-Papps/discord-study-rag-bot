from __future__ import annotations

from db.models import StatusReport
from rag.retriever import RAGAnswer
from rag.retriever import RetrievedChunk
from services.sync_service import SyncRunReport
from utils.text import shorten


DISCORD_MESSAGE_LIMIT = 1900
MAX_ANSWER_SOURCES = 3


def split_discord_message(text: str, limit: int = DISCORD_MESSAGE_LIMIT) -> list[str]:
    if len(text) <= limit:
        return [text]

    parts: list[str] = []
    remaining = text
    while len(remaining) > limit:
        split_at = remaining.rfind("\n", 0, limit)
        if split_at <= 0:
            split_at = limit
        parts.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def format_sync_report(report: SyncRunReport) -> str:
    lines = [
        f"{report.mode.title()} sync complete.",
        f"Messages seen: {report.total_messages_seen}",
        (
            "Messages stored/upserted: "
            f"{report.total_messages_stored} "
            f"(new={report.total_messages_inserted}, "
            f"updated={report.total_messages_updated}, "
            f"unchanged={report.total_messages_unchanged})"
        ),
        f"Chunks created: {report.indexing.chunks_created}",
        f"Chunks embedded: {report.indexing.chunks_embedded}",
    ]
    if report.indexing.channels_reindexed:
        lines.append(f"Channels reindexed after note edits: {report.indexing.channels_reindexed}")
    if report.indexing.stale_vectors_repaired:
        lines.append(f"Missing vectors repaired: {report.indexing.stale_vectors_repaired}")
    lines.extend(["", "Channels:"])
    for channel in report.channels:
        lines.append(
            f"- #{channel.channel_name}: seen={channel.messages_seen}, "
            f"new={channel.messages_inserted}, updated={channel.messages_updated}, "
            f"unchanged={channel.messages_unchanged}, "
            f"last_message={channel.last_message_id}, "
            f"source={channel.source_kind}"
        )
    return "\n".join(lines)


def format_status(report: StatusReport) -> str:
    lines = [
        "Local Discord RAG status",
        f"Messages: {report.total_messages}",
        f"Chunks: {report.total_chunks}",
        f"Embedded chunks: {report.embedded_chunks}",
        f"Vector store: {report.vector_count} vectors at {report.vector_path}",
        "",
        "Sync state:",
    ]
    if not report.sync_states:
        lines.append("- No channels synced yet.")
    for state in report.sync_states:
        lines.append(
            f"- #{state.channel_name} ({state.channel_id}): "
            f"last_message={state.last_synced_message_id}, "
            f"last_timestamp={state.last_synced_timestamp}, "
            f"last_full_sync={state.last_full_sync_at}"
        )
    return "\n".join(lines)


def format_rag_answer(answer: RAGAnswer) -> str:
    if answer.refused:
        return answer.answer

    lines = [
        "**Answer**",
        answer.answer.strip(),
    ]
    if answer.sources:
        lines.extend(["", "**Best sources**"])
        for index, source in enumerate(answer.sources[:MAX_ANSWER_SOURCES], start=1):
            source_name = (
                f"{source.category_name} / #{source.channel_name}"
                if source.category_name
                else f"#{source.channel_name}"
            )
            lines.append(
                f"{index}. {source_name}\n"
                f"   {source.timestamp}\n"
                f"   {source.message_url}\n"
                f"   `{shorten(source.snippet, 160)}`"
            )
    return "\n".join(lines)


def format_retrieved_chunks(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No chunks were retrieved above the current similarity threshold."

    lines = ["Retrieved chunks:"]
    for index, chunk in enumerate(chunks, start=1):
        lines.append(
            f"{index}. similarity={chunk.similarity:.3f} chunk={chunk.chunk_id}\n"
            f"{shorten(chunk.text, 500)}"
        )
    return "\n\n".join(lines)


def format_source_excerpts(source_label: str, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No synced chunks were found for that selected source."

    lines = [
        f"**Source excerpts from {source_label}**",
        "Relevant stored excerpts:",
    ]
    for index, chunk in enumerate(chunks[:5], start=1):
        lines.append(f"{index}. {shorten(chunk.text, 700)}")
    return "\n\n".join(lines)
