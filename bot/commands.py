from __future__ import annotations

import asyncio
import logging

import discord
from discord import app_commands
from discord.ext import commands

from bot.formatting import (
    format_rag_answer,
    format_retrieved_chunks,
    format_status,
    format_sync_report,
    split_discord_message,
)
from rag.answer_generator import REFUSAL_MESSAGE
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

    @bot.command(name="sync")
    async def sync_command(ctx: commands.Context) -> None:
        if not await ensure_allowed(ctx):
            return
        if sync_lock.locked():
            await ctx.reply("A sync is already running.", mention_author=False)
            return
        async with sync_lock:
            await ctx.reply("Starting full sync of configured channels.", mention_author=False)
            try:
                report = await context.sync_service.full_sync(bot)
            except RuntimeError as exc:
                logger.error("Full sync failed: %s", exc)
                await ctx.reply(str(exc), mention_author=False)
                return
            except Exception:
                logger.exception("Full sync failed")
                await ctx.reply("Full sync failed. Check local logs for details.", mention_author=False)
                return
            await send_chunks(ctx, format_sync_report(report))

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
                await interaction.followup.send(str(exc))
                return
            except Exception:
                logger.exception("Full sync failed")
                await interaction.followup.send("Full sync failed. Check local logs for details.")
                return
            await send_interaction_chunks(interaction, format_sync_report(report))

    @bot.command(name="resync")
    async def resync_command(ctx: commands.Context) -> None:
        if not await ensure_allowed(ctx):
            return
        if sync_lock.locked():
            await ctx.reply("A sync is already running.", mention_author=False)
            return
        async with sync_lock:
            await ctx.reply("Starting incremental sync.", mention_author=False)
            try:
                report = await context.sync_service.incremental_sync(bot)
            except RuntimeError as exc:
                logger.error("Incremental sync failed: %s", exc)
                await ctx.reply(str(exc), mention_author=False)
                return
            except Exception:
                logger.exception("Incremental sync failed")
                await ctx.reply(
                    "Incremental sync failed. Check local logs for details.",
                    mention_author=False,
                )
                return
            await send_chunks(ctx, format_sync_report(report))

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
                await interaction.followup.send(str(exc))
                return
            except Exception:
                logger.exception("Incremental sync failed")
                await interaction.followup.send(
                    "Incremental sync failed. Check local logs for details."
                )
                return
            await send_interaction_chunks(interaction, format_sync_report(report))

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
            answer = await asyncio.to_thread(context.rag_pipeline.answer_question, question)
        except Exception:
            logger.exception("Question answering failed")
            await interaction.followup.send(REFUSAL_MESSAGE)
            return
        await send_interaction_chunks(interaction, format_rag_answer(answer))

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
                "`/ask question:<question>` - answer from all synced notes",
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
