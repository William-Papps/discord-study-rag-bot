from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import discord

from db.models import StatusReport, StoredMessage
from db.repositories import ChunkRepository, MessageRepository, SyncStateRepository
from rag.vector_store import ChromaVectorStore
from services.attachment_text_service import AttachmentTextService
from services.indexing_service import IndexingReport, IndexingService
from utils.config import Settings
from utils.text import clean_discord_content

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChannelSyncReport:
    channel_id: int
    channel_name: str
    source_kind: str
    messages_seen: int
    messages_stored: int
    messages_inserted: int
    messages_updated: int
    messages_unchanged: int
    last_message_id: int | None


@dataclass(frozen=True)
class SyncTarget:
    channel_id: int
    channel_name: str
    category_name: str | None
    source_kind: str
    channel: discord.TextChannel | discord.Thread


@dataclass(frozen=True)
class SyncRunReport:
    mode: str
    channels: list[ChannelSyncReport]
    indexing: IndexingReport

    @property
    def total_messages_seen(self) -> int:
        return sum(channel.messages_seen for channel in self.channels)

    @property
    def total_messages_stored(self) -> int:
        return sum(channel.messages_stored for channel in self.channels)

    @property
    def total_messages_inserted(self) -> int:
        return sum(channel.messages_inserted for channel in self.channels)

    @property
    def total_messages_updated(self) -> int:
        return sum(channel.messages_updated for channel in self.channels)

    @property
    def total_messages_unchanged(self) -> int:
        return sum(channel.messages_unchanged for channel in self.channels)


class SyncService:
    def __init__(
        self,
        settings: Settings,
        message_repository: MessageRepository,
        chunk_repository: ChunkRepository,
        sync_state_repository: SyncStateRepository,
        indexing_service: IndexingService,
        vector_store: ChromaVectorStore,
        attachment_text_service: AttachmentTextService,
    ) -> None:
        self.settings = settings
        self.message_repository = message_repository
        self.chunk_repository = chunk_repository
        self.sync_state_repository = sync_state_repository
        self.indexing_service = indexing_service
        self.vector_store = vector_store
        self.attachment_text_service = attachment_text_service

    async def full_sync(self, bot: discord.Client) -> SyncRunReport:
        targets = await self._resolve_sync_targets(bot)
        reports = []
        for target in targets:
            reports.append(await self._sync_target(bot, target, full=True))
        indexing = await asyncio.to_thread(
            self._index_after_sync,
            [target.channel_id for target in targets],
            [report.channel_id for report in reports if report.messages_updated > 0],
        )
        return SyncRunReport(mode="full", channels=reports, indexing=indexing)

    async def incremental_sync(self, bot: discord.Client) -> SyncRunReport:
        targets = await self._resolve_sync_targets(bot)
        reports = []
        for target in targets:
            reports.append(await self._sync_target(bot, target, full=False))
        indexing = await asyncio.to_thread(
            self._index_after_sync,
            [target.channel_id for target in targets],
            [report.channel_id for report in reports if report.messages_updated > 0],
        )
        return SyncRunReport(mode="incremental", channels=reports, indexing=indexing)

    def _index_after_sync(
        self,
        channel_ids: list[int],
        changed_channel_ids: list[int],
    ) -> IndexingReport:
        changed = set(changed_channel_ids)
        unchanged_channel_ids = [channel_id for channel_id in channel_ids if channel_id not in changed]

        reindex_report = self.indexing_service.reindex_channels(list(changed))
        new_data_report = self.indexing_service.index_new_data(unchanged_channel_ids)
        return IndexingReport(
            chunks_created=reindex_report.chunks_created + new_data_report.chunks_created,
            chunks_embedded=reindex_report.chunks_embedded + new_data_report.chunks_embedded,
            channels_reindexed=reindex_report.channels_reindexed,
            stale_vectors_repaired=(
                reindex_report.stale_vectors_repaired
                + new_data_report.stale_vectors_repaired
            ),
        )

    async def _resolve_sync_targets(self, bot: discord.Client) -> list[SyncTarget]:
        targets = await self._targets_from_visible_guilds(bot)
        deduped: dict[int, SyncTarget] = {}
        for target in targets:
            if self._is_ignored_target(target):
                logger.info("Skipping ignored sync target #%s (%s)", target.channel_name, target.channel_id)
                continue
            deduped[target.channel_id] = target
        resolved = list(deduped.values())
        logger.info("Resolved %s sync targets", len(resolved))
        return resolved

    def _is_ignored_target(self, target: SyncTarget) -> bool:
        if target.channel_id in self.settings.discord_ignored_channel_ids:
            return True
        if isinstance(target.channel, discord.Thread):
            return target.channel.parent_id in self.settings.discord_ignored_channel_ids
        return False

    async def _targets_from_visible_guilds(self, bot: discord.Client) -> list[SyncTarget]:
        guilds = self._configured_guilds(bot)
        targets: list[SyncTarget] = []
        for guild in guilds:
            for channel in sorted(guild.channels, key=lambda item: (item.position, item.id)):
                if isinstance(channel, discord.CategoryChannel):
                    continue
                if isinstance(channel, discord.TextChannel):
                    targets.extend(await self._targets_from_text_channel(channel, guild.name))
                elif isinstance(channel, discord.ForumChannel):
                    targets.extend(
                        await self._targets_from_forum(
                            channel,
                            category_name=channel.category.name if channel.category else None,
                        )
                    )
        return targets

    def _configured_guilds(self, bot: discord.Client) -> list[discord.Guild]:
        if not self.settings.discord_guild_ids:
            return list(bot.guilds)

        guilds: list[discord.Guild] = []
        missing: list[int] = []
        for guild_id in self.settings.discord_guild_ids:
            guild = bot.get_guild(guild_id)
            if guild is None:
                missing.append(guild_id)
            else:
                guilds.append(guild)
        if missing:
            raise RuntimeError(f"Bot is not connected to configured guild IDs: {missing}")
        return guilds

    async def _targets_from_text_channel(
        self,
        channel: discord.TextChannel,
        guild_name: str,
    ) -> list[SyncTarget]:
        category_name = channel.category.name if channel.category else None
        targets = [self._make_target(channel, f"guild:{guild_name}", category_name)]
        for thread in channel.threads:
            targets.append(
                self._make_target(
                    thread,
                    f"guild:{guild_name}/thread:{channel.name}",
                    category_name,
                    parent_name=channel.name,
                )
            )
        async for thread in channel.archived_threads(limit=None):
            targets.append(
                self._make_target(
                    thread,
                    f"guild:{guild_name}/thread:{channel.name}",
                    category_name,
                    parent_name=channel.name,
                )
            )
        return targets

    async def _targets_from_forum(
        self,
        forum: discord.ForumChannel,
        category_name: str | None = None,
    ) -> list[SyncTarget]:
        targets: dict[int, SyncTarget] = {}
        source_kind = (
            f"category:{category_name}/forum:{forum.name}"
            if category_name
            else f"forum:{forum.name}"
        )

        for thread in forum.threads:
            targets[thread.id] = self._make_target(
                thread,
                source_kind,
                category_name,
                parent_name=forum.name,
            )

        async for thread in forum.archived_threads(limit=None):
            targets[thread.id] = self._make_target(
                thread,
                source_kind,
                category_name,
                parent_name=forum.name,
            )

        logger.info("Resolved %s forum post threads from #%s", len(targets), forum.name)
        return list(targets.values())

    @staticmethod
    def _make_target(
        channel: discord.TextChannel | discord.Thread,
        source_kind: str,
        category_name: str | None = None,
        parent_name: str | None = None,
    ) -> SyncTarget:
        channel_name = getattr(channel, "name", str(channel.id))
        if isinstance(channel, discord.Thread):
            parent = parent_name or getattr(channel.parent, "name", None)
            if parent:
                channel_name = f"{parent} / {channel.name}"
        return SyncTarget(
            channel_id=channel.id,
            channel_name=channel_name,
            category_name=category_name,
            source_kind=source_kind,
            channel=channel,
        )

    async def _sync_target(
        self,
        bot: discord.Client,
        target: SyncTarget,
        full: bool,
    ) -> ChannelSyncReport:
        channel = target.channel
        channel_id = target.channel_id
        channel_name = target.channel_name
        state = self.sync_state_repository.get(channel_id)
        after = None if full or state is None or state.last_synced_message_id is None else (
            discord.Object(id=state.last_synced_message_id)
        )

        logger.info(
            "Starting %s sync for #%s (%s)",
            "full" if full else "incremental",
            channel_name,
            channel_id,
        )

        batch: list[StoredMessage] = []
        messages_seen = 0
        messages_stored = 0
        messages_inserted = 0
        messages_updated = 0
        messages_unchanged = 0
        latest_message_id: int | None = state.last_synced_message_id if state else None
        latest_timestamp: str | None = state.last_synced_timestamp if state else None

        async for message in channel.history(
            limit=self.settings.discord_history_limit,
            oldest_first=True,
            after=after,
        ):
            if bot.user is not None and message.author.id == bot.user.id:
                continue
            raw_content = await self._raw_message_content(message)
            cleaned = clean_discord_content(raw_content)
            if not cleaned:
                continue

            stored_message = StoredMessage(
                message_id=message.id,
                channel_id=channel_id,
                channel_name=channel_name,
                category_name=target.category_name,
                author_id=message.author.id,
                author_name=str(message.author),
                timestamp=message.created_at.astimezone(timezone.utc).isoformat(),
                message_url=message.jump_url,
                raw_content=raw_content,
                cleaned_content=cleaned,
            )
            batch.append(stored_message)
            messages_seen += 1
            latest_message_id = message.id
            latest_timestamp = stored_message.timestamp

            if len(batch) >= 100:
                report = await asyncio.to_thread(
                    self.message_repository.upsert_many,
                    batch,
                )
                messages_stored += report.total
                messages_inserted += report.inserted
                messages_updated += report.updated
                messages_unchanged += report.unchanged
                batch = []

        if batch:
            report = await asyncio.to_thread(
                self.message_repository.upsert_many,
                batch,
            )
            messages_stored += report.total
            messages_inserted += report.inserted
            messages_updated += report.updated
            messages_unchanged += report.unchanged

        if latest_message_id is not None:
            last_full_sync_at = datetime.now(timezone.utc).isoformat() if full else None
            await asyncio.to_thread(
                self.sync_state_repository.upsert,
                channel_id,
                channel_name,
                latest_message_id,
                latest_timestamp,
                last_full_sync_at,
            )

        logger.info(
            "Finished sync for #%s: seen=%s inserted=%s updated=%s unchanged=%s last=%s",
            channel_name,
            messages_seen,
            messages_inserted,
            messages_updated,
            messages_unchanged,
            latest_message_id,
        )
        return ChannelSyncReport(
            channel_id=channel_id,
            channel_name=channel_name,
            messages_seen=messages_seen,
            messages_stored=messages_stored,
            messages_inserted=messages_inserted,
            messages_updated=messages_updated,
            messages_unchanged=messages_unchanged,
            last_message_id=latest_message_id,
            source_kind=target.source_kind,
        )

    def status(self) -> StatusReport:
        return StatusReport(
            total_messages=self.message_repository.count(),
            total_chunks=self.chunk_repository.count(),
            embedded_chunks=self.chunk_repository.count_embedded(),
            vector_count=self.vector_store.count(),
            vector_path=str(self.settings.chroma_path),
            sync_states=self.sync_state_repository.list_all(),
        )

    async def _raw_message_content(self, message: discord.Message) -> str:
        parts = [message.content or ""]
        for attachment in message.attachments:
            parts.append(f"[attachment: {attachment.filename}] {attachment.url}")
            extracted_text = await self.attachment_text_service.extract_text(attachment)
            if extracted_text:
                parts.append(
                    f"[image text from {attachment.filename}]\n{extracted_text}"
                )
        return "\n".join(part for part in parts if part).strip()
