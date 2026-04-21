# Local Discord RAG Assistant

This is a local-first Discord RAG assistant for a personal Discord server used as a school notes store. It syncs selected Discord categories, forum posts, threads, and text channels into local SQLite, chunks the stored notes, embeds them into a persistent Chroma vector store, and answers questions only from retrieved Discord note content.

If retrieval is weak or the notes do not support an answer, the bot returns exactly:

```text
This information is not found in your Discord notes.
```

## Project Structure

```text
main.py                    # application entrypoint
bot/                       # Discord bot setup, commands, response formatting
db/                        # SQLite connection, schema, repositories, dataclasses
rag/                       # chunking, embeddings, vector store, retrieval, answering
services/                  # startup, sync, indexing workflows
utils/                     # config, logging, Discord text cleanup
data/                      # local durable SQLite and Chroma storage
```

## Persistence

All important state is persisted under `data/` by default.

- `data/messages.db` stores raw Discord messages, cleaned content, chunk records, source mappings, sync state, and chunking progress.
- `data/chroma/` stores the persistent Chroma vector collection.
- `sync_state` tracks the last synced message per resolved text channel or forum post thread.
- `chunk_state` tracks the last message that has been chunked per channel.

The bot does not rebuild the vector database on startup. It loads SQLite and Chroma from disk, so previously indexed notes are available immediately after restart.

## Startup Behavior

On startup, `main.py`:

1. Loads `.env`.
2. Initializes logging.
3. Ensures local data directories exist.
4. Opens `data/messages.db`.
5. Creates the SQLite schema if needed.
6. Loads or creates the persistent Chroma collection.
7. Wires repositories, sync service, indexing service, retriever, and answer generator.
8. Starts the official Discord bot client.

Startup is idempotent. Existing messages, chunks, vectors, and sync cursors are reused.

## Shutdown Behavior

The bot runs locally until you stop it with `Ctrl+C` or close the terminal. SQLite is stored on disk and Chroma persists vectors under `data/chroma/`. On shutdown, the SQLite connection is closed. No important state is kept only in memory.

## Setup

1. Create a Discord application and bot in the Discord Developer Portal.
2. Invite the bot to your server with the `bot` and `applications.commands` OAuth2 scopes. Give it permission to view channels, read message history, send messages, and manage messages in the selected categories/channels.
3. Enable **Message Content Intent** in the Discord Developer Portal. Slash commands do not need it, but syncing your notes and attachments does.
4. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

5. Create `.env` from `.env.example`:

```powershell
Copy-Item .env.example .env
```

6. Fill in:

```text
DISCORD_BOT_TOKEN=your_bot_token
DISCORD_ENABLE_MESSAGE_CONTENT_INTENT=true
DISCORD_ENABLE_PREFIX_COMMANDS=false
DISCORD_SYNC_ALL_VISIBLE_CHANNELS=true
DISCORD_GUILD_IDS=your_server_id
DISCORD_IGNORED_CHANNEL_IDS=your_question_channel_id
OPENAI_API_KEY=your_openai_key
IMAGE_TEXT_PROVIDER=openai
```

The default workflow is:

```text
DISCORD_SYNC_ALL_VISIBLE_CHANNELS=true
DISCORD_IGNORED_CHANNEL_IDS=the_channel_where_you_use_slash_commands
```

With that setup, the bot syncs every text channel and forum post thread it can see, except the channel IDs listed in `DISCORD_IGNORED_CHANNEL_IDS`. Put your command/reply channel there so bot questions and answers do not get indexed as notes.

Set `DISCORD_GUILD_IDS` to your server ID while developing. Guild slash commands usually appear quickly. If you leave it blank, the bot registers global slash commands, which can take much longer to appear in Discord.

Use Discord permissions as the main boundary: if the bot can see a channel and it is not ignored, it can sync it.

For command access control, set:

```text
DISCORD_ALLOWED_USER_IDS=your_discord_user_id
```

Leave it blank to allow any user who can access the bot in your server.

## Running Locally

```powershell
python main.py
```

The bot will connect to Discord and be ready to answer from previously indexed local data. If this is the first run, use `/sync` in Discord to populate the local database and vector store.

If your notes are screenshots or photos, keep `IMAGE_TEXT_PROVIDER=openai`. During `/sync`, the bot transcribes supported image attachments and stores that text locally with the Discord message before chunking and embedding.

## Discord Commands

- `/sync`  
  Full sync of configured sources. It expands categories/forums into concrete text channels and forum post threads, fetches history, upserts messages into SQLite, chunks only not-yet-chunked local messages, embeds unembedded chunks, and persists all state.

- `/resync`  
  Incremental sync. It expands the same sources and fetches messages newer than the saved sync cursor for each resolved channel/thread.

- `/ask question:<question>`  
  Retrieves relevant chunks and answers only from those chunks.

- `/askchannel channel:#channel question:<question>`  
  Same as `/ask`, but filters retrieval to one channel.

- `/find query:<words> scope:<optional> limit:<number>`  
  Searches synced note text directly and returns exact matching snippets with source links. Scope can be a Discord category such as `DES502`, a forum/thread name, or a synced source ID. If omitted, all synced notes are searched.

- `/quiz scope:<category-or-source> topic:<optional> count:<number>`  
  Creates revision questions from exact synced note excerpts in one category, forum post, thread, or text channel. Answers are hidden behind Discord spoiler text and use exact source excerpts.

- `/flashcards scope:<category-or-source> topic:<optional> count:<number>`  
  Starts an interactive flashcard session. The bot asks a question based on an exact source excerpt; you submit your answer first, then it compares your answer to the source and privately reveals the exact source answer.

- `/summarize scope:<category-or-source> limit:<number>`  
  Shows an extractive summary made from direct note excerpts in that category/source, not rewritten AI prose.

- `/status`  
  Shows total messages, chunks, embedded chunks, vector count, vector path, and per-channel sync state.

- `/commands`  
  Shows a compact list of available bot commands and usage hints.

- `/purge amount:<number>`  
  Deletes that number of recent messages from the channel where the command is used. The amount must be between 1 and 100.

- `/reindex`  
  Deletes local chunk records and resets the Chroma collection, then rebuilds chunks and embeddings from all SQLite messages. It does not fetch Discord history.

## Retrieval and Answering

The answer pipeline:

1. Embeds the user question.
2. Queries Chroma for top matching chunks.
3. Filters results below `RETRIEVAL_MIN_SIMILARITY`.
4. Refuses if fewer than `RETRIEVAL_MIN_RESULTS` remain.
5. Sends only the retrieved Discord chunks to the answer provider.
6. Returns the answer plus source references.

Each answer source includes the channel name, timestamp, Discord message link, and a short snippet.

## Model Providers

The provider boundaries are in `rag/embeddings.py` and `rag/answer_generator.py`.

Default production-style settings:

```text
EMBEDDING_PROVIDER=openai
ANSWER_PROVIDER=openai
IMAGE_TEXT_PROVIDER=openai
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_ANSWER_MODEL=gpt-4o-mini
OPENAI_VISION_MODEL=gpt-4o-mini
```

For smoke tests without OpenAI calls:

```text
EMBEDDING_PROVIDER=local_hash
ANSWER_PROVIDER=extractive
IMAGE_TEXT_PROVIDER=none
```

`local_hash` is deterministic and persistent-compatible, but it is not a semantic embedding model. Use a real embedding provider for real retrieval quality.

Set `IMAGE_TEXT_PROVIDER=none` if you do not want image attachments sent to OpenAI for transcription. With OCR disabled, image-only Discord notes are stored as attachment links but their visible text is not searchable.

If sync fails with `does not have access to model`, your API key's OpenAI project does not have access to the model named in `.env`. Either change `OPENAI_EMBEDDING_MODEL`, `OPENAI_ANSWER_MODEL`, and `OPENAI_VISION_MODEL` to models available to that project, or use the local smoke-test settings above while you sort out model access.

## End-to-End Test

1. Add a few notes in one configured Discord category, forum post, or text channel, for example due dates or class instructions.
2. Start the bot:

```powershell
python main.py
```

3. In Discord, run:

```text
/sync
/status
/ask question:What is due next week?
```

4. Stop the bot with `Ctrl+C`.
5. Start it again:

```powershell
python main.py
```

6. Run the same question before syncing again:

```text
/ask question:What is due next week?
```

The bot should answer from previously persisted SQLite and Chroma data. If the notes do not contain the answer, it should return the required refusal sentence.

## Useful Tuning

- `RETRIEVAL_TOP_K`: how many chunks to retrieve.
- `RETRIEVAL_MIN_SIMILARITY`: raise this to make the bot refuse more often; lower it if `/debugretrieve` shows no chunks for questions you know are covered.
- `CHUNK_TARGET_TOKENS`: desired chunk size.
- `CHUNK_MAX_TOKENS`: hard chunk limit.
- `CHUNK_MAX_TIME_GAP_MINUTES`: split chunks when messages are too far apart.

## Notes

- This uses an official Discord bot account through `discord.py`. It is not a self-bot.
- The bot skips messages from bots to avoid indexing its own answers.
- Newly added class channels and forum posts are picked up the next time you run `/sync` or `/resync`, as long as the bot has permission to see them and they are not in `DISCORD_IGNORED_CHANNEL_IDS`.
- Legacy `!` commands still exist in the code, but slash commands are the intended interface.
- Image attachments are transcribed during sync when `IMAGE_TEXT_PROVIDER=openai`. Attachments are still stored with filename and URL for source traceability.
- Edited old messages are updated when fetched again. If the stored note text changes, that channel's chunks and vectors are rebuilt automatically during sync.
- Category scopes such as `DES502` are populated during `/sync`. If you synced data before category support was added, run `/sync` once so existing notes get their category metadata.
