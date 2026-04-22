from __future__ import annotations

import asyncio
import logging

import discord
from discord import app_commands
from discord.ext import commands

from bot.formatting import (
    format_rag_answer,
    format_retrieved_chunks,
    format_source_excerpts,
    format_status,
    split_discord_message,
)
from rag.answer_generator import REFUSAL_MESSAGE
from rag.answer_generator import SourceSelectionCandidate
from rag.retriever import RetrievedChunk
from services.startup_service import AppContext
from services.study_service import (
    StudyCard,
    format_find_results,
    format_flashcard,
    format_flashcard_answer,
    format_quiz,
    format_summary,
)

logger = logging.getLogger(__name__)

MAX_SELECTED_SOURCE_CHUNKS = 12


def log_sync_report(report: object) -> None:
    logger.info(
        "%s sync complete: channels=%s seen=%s stored=%s new=%s updated=%s unchanged=%s "
        "chunks_created=%s chunks_embedded=%s reindexed=%s repaired_vectors=%s",
        report.mode,
        len(report.channels),
        report.total_messages_seen,
        report.total_messages_stored,
        report.total_messages_inserted,
        report.total_messages_updated,
        report.total_messages_unchanged,
        report.indexing.chunks_created,
        report.indexing.chunks_embedded,
        report.indexing.channels_reindexed,
        report.indexing.stale_vectors_repaired,
    )


def create_bot(context: AppContext) -> commands.Bot:
    intents = discord.Intents.default()
    intents.message_content = context.settings.discord_enable_message_content_intent
    intents.guilds = True
    intents.messages = True
    app_command_guilds = [
        discord.Object(id=guild_id)
        for guild_id in context.settings.discord_guild_ids
    ] or None

    class LocalRAGBot(commands.Bot):
        async def setup_hook(self) -> None:
            if context.settings.discord_guild_ids:
                logger.info(
                    "Registering slash commands for configured guild IDs: %s",
                    context.settings.discord_guild_ids,
                )
                self.tree.clear_commands(guild=None)
                cleared = await self.tree.sync()
                logger.info("Cleared global slash commands; synced %s global commands", len(cleared))

                for guild_id in context.settings.discord_guild_ids:
                    guild = discord.Object(id=guild_id)
                    synced = await self.tree.sync(guild=guild)
                    logger.info("Synced %s slash commands to guild %s", len(synced), guild_id)
            else:
                synced = await self.tree.sync()
                logger.info("Synced %s global slash commands", len(synced))

        async def on_ready(self) -> None:
            logger.info("Discord bot connected as %s", self.user)

    bot = LocalRAGBot(
        command_prefix=context.settings.discord_command_prefix,
        intents=intents,
        help_command=None,
    )
    sync_lock = asyncio.Lock()

    async def send_chunks(ctx: commands.Context, text: str) -> None:
        for part in split_discord_message(text):
            await ctx.reply(part, mention_author=False)

    async def send_interaction_chunks(interaction: discord.Interaction, text: str) -> None:
        for part in split_discord_message(text):
            await interaction.followup.send(part)

    async def ensure_allowed(ctx: commands.Context) -> bool:
        allowed_ids = context.settings.discord_allowed_user_ids
        if allowed_ids and ctx.author.id not in allowed_ids:
            await ctx.reply("You are not allowed to use this bot.", mention_author=False)
            return False
        return True

    async def ensure_interaction_allowed(interaction: discord.Interaction) -> bool:
        allowed_ids = context.settings.discord_allowed_user_ids
        if allowed_ids and interaction.user.id not in allowed_ids:
            if interaction.response.is_done():
                await interaction.followup.send("You are not allowed to use this bot.", ephemeral=True)
            else:
                await interaction.response.send_message(
                    "You are not allowed to use this bot.",
                    ephemeral=True,
                )
            return False
        return True

    class FlashcardAnswerModal(discord.ui.Modal):
        def __init__(self, view: "FlashcardView", card: StudyCard) -> None:
            super().__init__(title="Flashcard answer")
            self.flashcard_view = view
            self.card = card
            self.answer = discord.ui.TextInput(
                label="Your answer",
                style=discord.TextStyle.paragraph,
                required=True,
                max_length=1000,
            )
            self.add_item(self.answer)

        async def on_submit(self, interaction: discord.Interaction) -> None:
            await self.flashcard_view.submit_answer(
                interaction,
                str(self.answer.value),
            )

    class FlashcardView(discord.ui.View):
        def __init__(self, user_id: int, cards: list[StudyCard]) -> None:
            super().__init__(timeout=900)
            self.user_id = user_id
            self.cards = cards
            self.index = 0
            self.message: discord.WebhookMessage | discord.Message | None = None

        @discord.ui.button(label="Submit answer", style=discord.ButtonStyle.primary)
        async def submit_button(
            self,
            interaction: discord.Interaction,
            button: discord.ui.Button,
        ) -> None:
            if interaction.user.id != self.user_id:
                await interaction.response.send_message(
                    "This flashcard session belongs to another user.",
                    ephemeral=True,
                )
                return
            await interaction.response.send_modal(
                FlashcardAnswerModal(self, self.cards[self.index])
            )

        async def submit_answer(
            self,
            interaction: discord.Interaction,
            user_answer: str,
        ) -> None:
            card = self.cards[self.index]
            feedback = await asyncio.to_thread(
                context.study_service.evaluate_flashcard_answer,
                card,
                user_answer,
            )
            await interaction.response.send_message(
                format_flashcard_answer(card, user_answer, feedback),
                ephemeral=True,
            )

            self.index += 1
            if self.message is None:
                return
            if self.index >= len(self.cards):
                for child in self.children:
                    if isinstance(child, discord.ui.Button):
                        child.disabled = True
                await self.message.edit(content="Flashcards complete.", view=self)
                self.stop()
                return

            await self.message.edit(
                content=format_flashcard(
                    self.cards[self.index],
                    self.index + 1,
                    len(self.cards),
                ),
                view=self,
            )

    class AskSourceSelect(discord.ui.Select):
        def __init__(
            self,
            question: str,
            grouped_chunks: dict[str, list[RetrievedChunk]],
            labels: dict[str, str],
        ) -> None:
            self.question = question
            self.grouped_chunks = grouped_chunks
            self.labels = labels
            options = [
                discord.SelectOption(
                    label=labels[key][:100],
                    value=key,
                    description=f"{len(grouped_chunks[key])} matching chunks",
                )
                for key in grouped_chunks
            ][:25]
            super().__init__(
                placeholder="Choose which source to answer from",
                min_values=1,
                max_values=1,
                options=options,
            )

        async def callback(self, interaction: discord.Interaction) -> None:
            view = self.view
            if not isinstance(view, AskDisambiguationView):
                await interaction.response.send_message("This selection expired.", ephemeral=True)
                return
            if interaction.user.id != view.user_id:
                await interaction.response.send_message(
                    "This source selection belongs to another user.",
                    ephemeral=True,
                )
                return

            selected = self.values[0]
            selected_chunks = self.grouped_chunks[selected]
            await interaction.response.defer(thinking=True)
            chunks = await asyncio.to_thread(
                expand_selected_source_chunks,
                int(selected),
                selected_chunks,
            )
            answer = await asyncio.to_thread(
                context.rag_pipeline.answer_from_chunks,
                self.question,
                chunks,
                False,
            )
            for child in view.children:
                if isinstance(child, discord.ui.Select):
                    child.disabled = True
            await interaction.message.edit(
                content=f"Selected source: {self.labels[selected]}",
                view=view,
            )
            if answer.refused:
                await send_interaction_chunks(
                    interaction,
                    format_source_excerpts(self.labels[selected], chunks),
                )
            else:
                await send_interaction_chunks(interaction, format_rag_answer(answer))
            view.stop()

    class AskDisambiguationView(discord.ui.View):
        def __init__(
            self,
            user_id: int,
            question: str,
            grouped_chunks: dict[str, list[RetrievedChunk]],
            labels: dict[str, str],
        ) -> None:
            super().__init__(timeout=180)
            self.user_id = user_id
            self.add_item(AskSourceSelect(question, grouped_chunks, labels))

    async def answer_or_select_source(
        interaction: discord.Interaction,
        question: str,
        channel_id: int | None = None,
    ) -> None:
        retrieved = await asyncio.to_thread(
            context.rag_pipeline.retriever.retrieve,
            question,
            channel_id,
        )
        grouped_chunks, labels = await asyncio.to_thread(
            build_source_options,
            question,
            retrieved,
        )
        if len(grouped_chunks) >= 2:
            view = AskDisambiguationView(
                user_id=interaction.user.id,
                question=question,
                grouped_chunks=grouped_chunks,
                labels=labels,
            )
            await interaction.followup.send(
                "I found multiple strong matching sources. Choose which one to answer from:",
                view=view,
            )
            return

        answer = await asyncio.to_thread(
            context.rag_pipeline.answer_from_chunks,
            question,
            retrieved,
        )
        if answer.refused and retrieved:
            await send_interaction_chunks(
                interaction,
                format_source_excerpts("retrieved notes", retrieved),
            )
        else:
            await send_interaction_chunks(interaction, format_rag_answer(answer))

    def build_source_options(
        question: str,
        retrieved: list[RetrievedChunk],
    ) -> tuple[dict[str, list[RetrievedChunk]], dict[str, str]]:
        if len(retrieved) < 2:
            return {}, {}
        references = context.chunk_repository.get_source_references(
            [chunk.chunk_id for chunk in retrieved]
        )
        references_by_chunk = {reference.chunk_id: reference for reference in references}
        groups: dict[str, list[RetrievedChunk]] = {}
        labels: dict[str, str] = {}
        best_similarity: dict[str, float] = {}
        top_similarity = retrieved[0].similarity if retrieved else 0.0
        for chunk in retrieved:
            if chunk.similarity < max(context.settings.retrieval_min_similarity, top_similarity - 0.08):
                continue
            reference = references_by_chunk.get(chunk.chunk_id)
            if reference is None:
                continue
            key = str(reference.channel_id)
            label = (
                f"{reference.category_name} / #{reference.channel_name}"
                if reference.category_name
                else f"#{reference.channel_name}"
            )
            groups.setdefault(key, []).append(chunk)
            labels[key] = label
            best_similarity[key] = max(best_similarity.get(key, 0.0), chunk.similarity)

        if len(groups) < 2:
            return {}, {}

        candidates = [
            SourceSelectionCandidate(
                source_id=key,
                label=labels[key],
                snippet="\n\n".join(chunk.text for chunk in chunks[:2]),
                similarity=best_similarity[key],
            )
            for key, chunks in groups.items()
        ]
        try:
            selected_source_ids = context.rag_pipeline.answer_generator.choose_competing_sources(
                question,
                candidates,
            )
        except Exception:
            logger.exception("Source ambiguity comparison failed")
            return {}, {}
        if len(selected_source_ids) < 2:
            return {}, {}

        selected_groups = {
            source_id: groups[source_id]
            for source_id in selected_source_ids
            if source_id in groups
        }
        selected_labels = {
            source_id: labels[source_id]
            for source_id in selected_groups
        }
        if len(selected_groups) < 2:
            return {}, {}
        return selected_groups, selected_labels

    def expand_selected_source_chunks(
        channel_id: int,
        selected_chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        expanded: list[RetrievedChunk] = []
        seen_chunk_ids: set[str] = set()

        for chunk in selected_chunks:
            expanded.append(chunk)
            seen_chunk_ids.add(chunk.chunk_id)
            if len(expanded) >= MAX_SELECTED_SOURCE_CHUNKS:
                return expanded

        for chunk in context.chunk_repository.list_for_channel(channel_id):
            if chunk.chunk_id in seen_chunk_ids:
                continue
            expanded.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.chunk_text,
                    similarity=0.0,
                )
            )
            seen_chunk_ids.add(chunk.chunk_id)
            if len(expanded) >= MAX_SELECTED_SOURCE_CHUNKS:
                break

        return expanded

    @bot.command(name="sync")
    async def sync_command(ctx: commands.Context) -> None:
        if not await ensure_allowed(ctx):
            return
        if sync_lock.locked():
            await ctx.reply("A sync is already running.", mention_author=False)
            return
        async with sync_lock:
            try:
                report = await context.sync_service.full_sync(bot)
            except RuntimeError as exc:
                logger.error("Full sync failed: %s", exc)
                await ctx.reply("Full sync failed. Check local logs for details.", mention_author=False)
                return
            except Exception:
                logger.exception("Full sync failed")
                await ctx.reply("Full sync failed. Check local logs for details.", mention_author=False)
                return
            log_sync_report(report)
            await ctx.reply("Full sync completed.", mention_author=False)

    @bot.tree.command(
        name="sync",
        description="Full sync configured Discord note sources.",
        guilds=app_command_guilds,
    )
    async def sync_slash(interaction: discord.Interaction) -> None:
        if not await ensure_interaction_allowed(interaction):
            return
        await interaction.response.defer(thinking=True)
        if sync_lock.locked():
            await interaction.followup.send("A sync is already running.")
            return
        async with sync_lock:
            try:
                report = await context.sync_service.full_sync(bot)
            except RuntimeError as exc:
                logger.error("Full sync failed: %s", exc)
                await interaction.followup.send("Full sync failed. Check local logs for details.")
                return
            except Exception:
                logger.exception("Full sync failed")
                await interaction.followup.send("Full sync failed. Check local logs for details.")
                return
            log_sync_report(report)
            await interaction.followup.send("Full sync completed.")

    @bot.command(name="resync")
    async def resync_command(ctx: commands.Context) -> None:
        if not await ensure_allowed(ctx):
            return
        if sync_lock.locked():
            await ctx.reply("A sync is already running.", mention_author=False)
            return
        async with sync_lock:
            try:
                report = await context.sync_service.incremental_sync(bot)
            except RuntimeError as exc:
                logger.error("Incremental sync failed: %s", exc)
                await ctx.reply(
                    "Incremental sync failed. Check local logs for details.",
                    mention_author=False,
                )
                return
            except Exception:
                logger.exception("Incremental sync failed")
                await ctx.reply(
                    "Incremental sync failed. Check local logs for details.",
                    mention_author=False,
                )
                return
            log_sync_report(report)
            await ctx.reply("Incremental sync completed.", mention_author=False)

    @bot.tree.command(
        name="resync",
        description="Incrementally sync new Discord notes.",
        guilds=app_command_guilds,
    )
    async def resync_slash(interaction: discord.Interaction) -> None:
        if not await ensure_interaction_allowed(interaction):
            return
        await interaction.response.defer(thinking=True)
        if sync_lock.locked():
            await interaction.followup.send("A sync is already running.")
            return
        async with sync_lock:
            try:
                report = await context.sync_service.incremental_sync(bot)
            except RuntimeError as exc:
                logger.error("Incremental sync failed: %s", exc)
                await interaction.followup.send(
                    "Incremental sync failed. Check local logs for details."
                )
                return
            except Exception:
                logger.exception("Incremental sync failed")
                await interaction.followup.send(
                    "Incremental sync failed. Check local logs for details."
                )
                return
            log_sync_report(report)
            await interaction.followup.send("Incremental sync completed.")

    @bot.command(name="ask")
    async def ask_command(ctx: commands.Context, *, question: str = "") -> None:
        if not await ensure_allowed(ctx):
            return
        question = question.strip()
        if not question:
            await ctx.reply("Usage: !ask <question>", mention_author=False)
            return
        try:
            answer = await asyncio.to_thread(context.rag_pipeline.answer_question, question)
        except Exception:
            logger.exception("Question answering failed")
            await ctx.reply(REFUSAL_MESSAGE, mention_author=False)
            return
        if answer.refused and answer.retrieved_chunks:
            await send_chunks(
                ctx,
                format_source_excerpts("retrieved notes", answer.retrieved_chunks),
            )
        else:
            await send_chunks(ctx, format_rag_answer(answer))

    @bot.tree.command(
        name="ask",
        description="Ask a question using only synced Discord notes.",
        guilds=app_command_guilds,
    )
    @app_commands.describe(question="Question to answer from your synced Discord notes")
    async def ask_slash(interaction: discord.Interaction, question: str) -> None:
        if not await ensure_interaction_allowed(interaction):
            return
        await interaction.response.defer(thinking=True)
        question = question.strip()
        if not question:
            await interaction.followup.send("Question cannot be empty.")
            return
        try:
            await answer_or_select_source(interaction, question)
        except Exception:
            logger.exception("Question answering failed")
            await interaction.followup.send(REFUSAL_MESSAGE)
            return

    @bot.tree.command(
        name="debugretrieve",
        description="Show retrieved note chunks for a question without generating an answer.",
        guilds=app_command_guilds,
    )
    @app_commands.describe(question="Question to retrieve matching note chunks for")
    async def debug_retrieve_slash(interaction: discord.Interaction, question: str) -> None:
        if not await ensure_interaction_allowed(interaction):
            return
        await interaction.response.defer(thinking=True, ephemeral=True)
        chunks = await asyncio.to_thread(context.rag_pipeline.retriever.retrieve, question.strip())
        await send_interaction_chunks(interaction, format_retrieved_chunks(chunks))

    @bot.command(name="askchannel")
    async def ask_channel_command(
        ctx: commands.Context,
        channel: discord.TextChannel,
        *,
        question: str = "",
    ) -> None:
        if not await ensure_allowed(ctx):
            return
        question = question.strip()
        if not question:
            await ctx.reply("Usage: !askchannel #channel <question>", mention_author=False)
            return
        try:
            answer = await asyncio.to_thread(
                context.rag_pipeline.answer_question,
                question,
                channel.id,
            )
        except Exception:
            logger.exception("Channel question answering failed")
            await ctx.reply(REFUSAL_MESSAGE, mention_author=False)
            return
        if answer.refused and answer.retrieved_chunks:
            await send_chunks(
                ctx,
                format_source_excerpts(f"#{channel.name}", answer.retrieved_chunks),
            )
        else:
            await send_chunks(ctx, format_rag_answer(answer))

    @bot.tree.command(
        name="askchannel",
        description="Ask a question using only notes from one text channel.",
        guilds=app_command_guilds,
    )
    @app_commands.describe(
        channel="Discord text channel to search",
        question="Question to answer from that channel's synced notes",
    )
    async def ask_channel_slash(
        interaction: discord.Interaction,
        channel: discord.TextChannel,
        question: str,
    ) -> None:
        if not await ensure_interaction_allowed(interaction):
            return
        await interaction.response.defer(thinking=True)
        question = question.strip()
        if not question:
            await interaction.followup.send("Question cannot be empty.")
            return
        try:
            answer = await asyncio.to_thread(
                context.rag_pipeline.answer_question,
                question,
                channel.id,
            )
        except Exception:
            logger.exception("Channel question answering failed")
            await interaction.followup.send(REFUSAL_MESSAGE)
            return
        if answer.refused and answer.retrieved_chunks:
            await send_interaction_chunks(
                interaction,
                format_source_excerpts(f"#{channel.name}", answer.retrieved_chunks),
            )
        else:
            await send_interaction_chunks(interaction, format_rag_answer(answer))

    @bot.tree.command(
        name="find",
        description="Find exact synced note snippets by keyword.",
        guilds=app_command_guilds,
    )
    @app_commands.describe(
        query="Words to search for in synced notes",
        scope="Optional category/thread/forum name, for example DES502",
        limit="Maximum results to show, from 1 to 15",
    )
    async def find_slash(
        interaction: discord.Interaction,
        query: str,
        scope: str = "",
        limit: app_commands.Range[int, 1, 15] = 8,
    ) -> None:
        if not await ensure_interaction_allowed(interaction):
            return
        await interaction.response.defer(thinking=True)
        query = query.strip()
        if not query:
            await interaction.followup.send("Query cannot be empty.")
            return

        try:
            results = await asyncio.to_thread(
                context.study_service.find,
                query,
                scope,
                int(limit),
            )
        except ValueError as exc:
            await interaction.followup.send(str(exc))
            return
        await send_interaction_chunks(interaction, format_find_results(results))

    @bot.tree.command(
        name="quiz",
        description="Create source-grounded revision questions from a category or source.",
        guilds=app_command_guilds,
    )
    @app_commands.describe(
        scope="Category/thread/forum name, for example DES502",
        topic="Optional topic or keyword to focus on",
        count="Number of questions, from 1 to 10",
    )
    async def quiz_slash(
        interaction: discord.Interaction,
        scope: str,
        topic: str = "",
        count: app_commands.Range[int, 1, 10] = 5,
    ) -> None:
        if not await ensure_interaction_allowed(interaction):
            return
        await interaction.response.defer(thinking=True)
        try:
            _, cards = await asyncio.to_thread(
                context.study_service.make_quiz,
                scope,
                topic.strip(),
                int(count),
            )
        except ValueError as exc:
            await interaction.followup.send(str(exc))
            return
        await send_interaction_chunks(interaction, format_quiz(cards))

    @bot.tree.command(
        name="summarize",
        description="Show direct note excerpts from a category or source.",
        guilds=app_command_guilds,
    )
    @app_commands.describe(
        scope="Category/thread/forum name, for example DES502",
        limit="Number of direct excerpts to include, from 1 to 25",
    )
    async def summarize_slash(
        interaction: discord.Interaction,
        scope: str,
        limit: app_commands.Range[int, 1, 25] = 12,
    ) -> None:
        if not await ensure_interaction_allowed(interaction):
            return
        await interaction.response.defer(thinking=True)
        try:
            resolved_scope, cards = await asyncio.to_thread(
                context.study_service.summarize_scope,
                scope,
                int(limit),
            )
        except ValueError as exc:
            await interaction.followup.send(str(exc))
            return
        await send_interaction_chunks(
            interaction,
            format_summary(cards, resolved_scope.label),
        )

    @bot.tree.command(
        name="flashcards",
        description="Start interactive source-grounded flashcards from a category or source.",
        guilds=app_command_guilds,
    )
    @app_commands.describe(
        scope="Category/thread/forum name, for example DES502",
        topic="Optional topic or keyword to focus on",
        count="Number of flashcards, from 1 to 10",
    )
    async def flashcards_slash(
        interaction: discord.Interaction,
        scope: str,
        topic: str = "",
        count: app_commands.Range[int, 1, 10] = 5,
    ) -> None:
        if not await ensure_interaction_allowed(interaction):
            return
        await interaction.response.defer(thinking=True)
        try:
            _, cards = await asyncio.to_thread(
                context.study_service.make_flashcards,
                scope,
                topic.strip(),
                int(count),
            )
        except ValueError as exc:
            await interaction.followup.send(str(exc))
            return
        if not cards:
            await interaction.followup.send("No synced notes were found for that flashcard request.")
            return

        view = FlashcardView(user_id=interaction.user.id, cards=cards)
        message = await interaction.followup.send(
            format_flashcard(cards[0], 1, len(cards)),
            view=view,
            wait=True,
        )
        view.message = message

    @bot.command(name="status")
    async def status_command(ctx: commands.Context) -> None:
        if not await ensure_allowed(ctx):
            return
        report = await asyncio.to_thread(context.sync_service.status)
        await send_chunks(ctx, format_status(report))

    @bot.tree.command(
        name="status",
        description="Show local Discord RAG storage status.",
        guilds=app_command_guilds,
    )
    async def status_slash(interaction: discord.Interaction) -> None:
        if not await ensure_interaction_allowed(interaction):
            return
        await interaction.response.defer(thinking=True)
        report = await asyncio.to_thread(context.sync_service.status)
        await send_interaction_chunks(interaction, format_status(report))

    @bot.tree.command(
        name="commands",
        description="Show available bot commands.",
        guilds=app_command_guilds,
    )
    async def commands_slash(interaction: discord.Interaction) -> None:
        if not await ensure_interaction_allowed(interaction):
            return
        await interaction.response.defer(thinking=True, ephemeral=True)
        text = "\n".join(
            [
                "**Commands**",
                "`/sync` - full sync visible note sources",
                "`/resync` - sync only new Discord notes",
                "`/ask question:<question>` - answer from notes; may ask you to pick a matching source",
                "`/askchannel channel:#channel question:<question>` - answer from one text channel",
                "`/find query:<words> scope:<optional>` - exact keyword search",
                "`/quiz scope:<category-or-source> topic:<optional>` - source-grounded quiz",
                "`/flashcards scope:<category-or-source> topic:<optional>` - interactive flashcards",
                "`/summarize scope:<category-or-source>` - direct excerpt summary",
                "`/debugretrieve question:<question>` - inspect retrieved chunks",
                "`/status` - show local storage/index status",
                "`/purge amount:<1-100>` - delete recent messages in this channel",
                "`/reindex` - rebuild chunks and vectors from local SQLite",
                "",
                "For `scope`, use a category like `DES502`, a forum/thread name, or a synced source ID.",
            ]
        )
        await send_interaction_chunks(interaction, text)

    @bot.tree.command(
        name="purge",
        description="Delete recent messages from the current channel.",
        guilds=app_command_guilds,
    )
    @app_commands.describe(amount="Number of recent messages to delete, from 1 to 100")
    async def purge_slash(
        interaction: discord.Interaction,
        amount: app_commands.Range[int, 1, 100],
    ) -> None:
        if not await ensure_interaction_allowed(interaction):
            return
        await interaction.response.defer(thinking=True, ephemeral=True)

        channel = interaction.channel
        if channel is None or not hasattr(channel, "purge"):
            await interaction.followup.send("This channel does not support message purging.")
            return

        try:
            deleted = await channel.purge(
                limit=int(amount),
                reason=f"Requested by {interaction.user}",
            )
        except discord.Forbidden:
            await interaction.followup.send(
                "I do not have permission to delete messages in this channel."
            )
            return
        except discord.HTTPException:
            logger.exception("Discord purge failed")
            await interaction.followup.send("Purge failed. Check local logs for details.")
            return

        deleted_message_ids = [message.id for message in deleted]
        deleted_local = await asyncio.to_thread(
            context.message_repository.delete_many,
            deleted_message_ids,
        )
        if deleted_local:
            await asyncio.to_thread(
                context.indexing_service.reindex_channels,
                [channel.id],
            )

        response = f"Deleted {len(deleted)} messages."
        if deleted_local:
            response += f" Removed {deleted_local} synced messages from local search."
        await interaction.followup.send(response)

    @bot.command(name="reindex")
    async def reindex_command(ctx: commands.Context) -> None:
        if not await ensure_allowed(ctx):
            return
        if sync_lock.locked():
            await ctx.reply("A sync or reindex is already running.", mention_author=False)
            return
        async with sync_lock:
            await ctx.reply(
                "Rebuilding chunks and vectors from local SQLite messages.",
                mention_author=False,
            )
            try:
                report = await asyncio.to_thread(
                    context.indexing_service.reindex_all_local_messages,
                )
            except RuntimeError as exc:
                logger.error("Reindex failed: %s", exc)
                await ctx.reply(str(exc), mention_author=False)
                return
            except Exception:
                logger.exception("Reindex failed")
                await ctx.reply("Reindex failed. Check local logs for details.", mention_author=False)
                return
            await ctx.reply(
                f"Reindex complete. Chunks created: {report.chunks_created}. "
                f"Chunks embedded: {report.chunks_embedded}.",
                mention_author=False,
            )

    @bot.tree.command(
        name="reindex",
        description="Rebuild chunks and vectors from local SQLite messages.",
        guilds=app_command_guilds,
    )
    async def reindex_slash(interaction: discord.Interaction) -> None:
        if not await ensure_interaction_allowed(interaction):
            return
        await interaction.response.defer(thinking=True)
        if sync_lock.locked():
            await interaction.followup.send("A sync or reindex is already running.")
            return
        async with sync_lock:
            try:
                report = await asyncio.to_thread(
                    context.indexing_service.reindex_all_local_messages,
                )
            except RuntimeError as exc:
                logger.error("Reindex failed: %s", exc)
                await interaction.followup.send(str(exc))
                return
            except Exception:
                logger.exception("Reindex failed")
                await interaction.followup.send("Reindex failed. Check local logs for details.")
                return
            await interaction.followup.send(
                f"Reindex complete. Chunks created: {report.chunks_created}. "
                f"Chunks embedded: {report.chunks_embedded}."
            )

    @bot.command(name="help")
    async def help_command(ctx: commands.Context) -> None:
        if not await ensure_allowed(ctx):
            return
        await ctx.reply(
            "\n".join(
                [
                    "Commands:",
                    "!sync - full sync configured channels",
                    "!resync - incremental sync only",
                    "!ask <question> - ask across synced notes",
                    "!askchannel #channel <question> - ask within one channel",
                    "!status - show local storage and sync state",
                    "!reindex - rebuild local chunks and vectors from SQLite",
                    "",
                    "Slash commands are also available: /sync, /resync, /ask,",
                    "/askchannel, /find, /quiz, /summarize, /flashcards,",
                    "/debugretrieve, /status, /commands, /purge, /reindex.",
                ]
            ),
            mention_author=False,
        )

    @bot.tree.error
    async def on_app_command_error(
        interaction: discord.Interaction,
        error: app_commands.AppCommandError,
    ) -> None:
        logger.exception("Discord slash command failed: %s", error)
        message = "Command failed. Check local logs for details."
        if interaction.response.is_done():
            await interaction.followup.send(message, ephemeral=True)
        else:
            await interaction.response.send_message(message, ephemeral=True)

    @bot.event
    async def on_command_error(ctx: commands.Context, error: commands.CommandError) -> None:
        if isinstance(error, commands.CommandNotFound):
            return
        if isinstance(error, commands.BadArgument):
            await ctx.reply(f"Invalid command arguments: {error}", mention_author=False)
            return
        logger.exception("Discord command failed: %s", error)
        await ctx.reply("Command failed. Check local logs for details.", mention_author=False)

    if not context.settings.discord_enable_prefix_commands:
        for name in ("sync", "resync", "ask", "askchannel", "status", "reindex", "help"):
            bot.remove_command(name)

    return bot
