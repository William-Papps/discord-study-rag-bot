from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv


def _parse_int_list(raw: str | None) -> list[int]:
    if not raw:
        return []
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if item in {"...", "replace_me", "your_server_id", "your_question_channel_id"}:
            continue
        try:
            values.append(int(item))
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid Discord ID value {item!r}. Use numeric IDs separated by commas."
            ) from exc
    return values


def _parse_optional_int_set(raw: str | None) -> set[int]:
    return set(_parse_int_list(raw))


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    discord_bot_token: str
    discord_sync_all_visible_channels: bool
    discord_guild_ids: list[int]
    discord_ignored_channel_ids: set[int]
    discord_enable_message_content_intent: bool
    discord_enable_prefix_commands: bool
    discord_command_prefix: str
    discord_allowed_user_ids: set[int]

    database_path: Path
    chroma_path: Path
    chroma_collection: str

    retrieval_top_k: int
    retrieval_min_similarity: float
    retrieval_min_results: int

    chunk_target_tokens: int
    chunk_max_tokens: int
    chunk_overlap_messages: int
    chunk_max_time_gap_minutes: int

    discord_history_limit: int | None

    embedding_provider: str
    answer_provider: str
    image_text_provider: str
    openai_api_key: str | None
    openai_project_id: str | None
    openai_embedding_model: str
    openai_answer_model: str
    openai_vision_model: str
    max_attachment_bytes: int

    log_level: str

    def ensure_directories(self) -> None:
        for path in (self.database_path.parent, self.chroma_path):
            path.mkdir(parents=True, exist_ok=True)


def load_settings(env_file: str | Path | None = None) -> Settings:
    load_dotenv(env_file, override=True)

    token = os.getenv("DISCORD_BOT_TOKEN", "").strip()
    sync_all_visible = _bool_env("DISCORD_SYNC_ALL_VISIBLE_CHANNELS", True)
    guild_ids = _parse_int_list(os.getenv("DISCORD_GUILD_IDS"))
    ignored_channel_ids = _parse_optional_int_set(os.getenv("DISCORD_IGNORED_CHANNEL_IDS"))
    history_limit = _int_env("DISCORD_HISTORY_LIMIT", 0)

    settings = Settings(
        discord_bot_token=token,
        discord_sync_all_visible_channels=sync_all_visible,
        discord_guild_ids=guild_ids,
        discord_ignored_channel_ids=ignored_channel_ids,
        discord_enable_message_content_intent=_bool_env(
            "DISCORD_ENABLE_MESSAGE_CONTENT_INTENT",
            True,
        ),
        discord_enable_prefix_commands=_bool_env("DISCORD_ENABLE_PREFIX_COMMANDS", False),
        discord_command_prefix=os.getenv("DISCORD_COMMAND_PREFIX", "!").strip() or "!",
        discord_allowed_user_ids=_parse_optional_int_set(os.getenv("DISCORD_ALLOWED_USER_IDS")),
        database_path=Path(os.getenv("DATABASE_PATH", "data/messages.db")),
        chroma_path=Path(os.getenv("CHROMA_PATH", "data/chroma")),
        chroma_collection=os.getenv("CHROMA_COLLECTION", "discord_notes").strip() or "discord_notes",
        retrieval_top_k=_int_env("RETRIEVAL_TOP_K", 5),
        retrieval_min_similarity=_float_env("RETRIEVAL_MIN_SIMILARITY", 0.35),
        retrieval_min_results=_int_env("RETRIEVAL_MIN_RESULTS", 1),
        chunk_target_tokens=_int_env("CHUNK_TARGET_TOKENS", 500),
        chunk_max_tokens=_int_env("CHUNK_MAX_TOKENS", 800),
        chunk_overlap_messages=_int_env("CHUNK_OVERLAP_MESSAGES", 1),
        chunk_max_time_gap_minutes=_int_env("CHUNK_MAX_TIME_GAP_MINUTES", 120),
        discord_history_limit=None if history_limit <= 0 else history_limit,
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower(),
        answer_provider=os.getenv("ANSWER_PROVIDER", "openai").strip().lower(),
        image_text_provider=os.getenv("IMAGE_TEXT_PROVIDER", "openai").strip().lower(),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_project_id=_optional_env("OPENAI_PROJECT_ID"),
        openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip(),
        openai_answer_model=os.getenv("OPENAI_ANSWER_MODEL", "gpt-4o-mini").strip(),
        openai_vision_model=os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini").strip(),
        max_attachment_bytes=_int_env("MAX_ATTACHMENT_BYTES", 5_000_000),
        log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
    )

    _validate(settings)
    return settings


def _validate(settings: Settings) -> None:
    missing: list[str] = []
    if not settings.discord_bot_token:
        missing.append("DISCORD_BOT_TOKEN")
    if not settings.discord_sync_all_visible_channels:
        missing.append("DISCORD_SYNC_ALL_VISIBLE_CHANNELS=true")
    if settings.embedding_provider == "openai" and not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if settings.answer_provider == "openai" and not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if settings.image_text_provider == "openai" and not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if missing:
        unique = sorted(set(missing))
        raise RuntimeError(f"Missing required configuration: {', '.join(unique)}")


def _optional_env(name: str) -> str | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    return raw.strip()
