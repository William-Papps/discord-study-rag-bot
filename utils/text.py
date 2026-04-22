from __future__ import annotations

import hashlib
import re
from datetime import datetime

try:
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None


MENTION_RE = re.compile(r"<@!?(\d+)>")
ROLE_RE = re.compile(r"<@&(\d+)>")
CHANNEL_RE = re.compile(r"<#(\d+)>")
CUSTOM_EMOJI_RE = re.compile(r"<a?:([A-Za-z0-9_]+):\d+>")
WHITESPACE_RE = re.compile(r"[ \t]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def clean_discord_content(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = MENTION_RE.sub(r"@user-\1", text)
    text = ROLE_RE.sub(r"@role-\1", text)
    text = CHANNEL_RE.sub(r"#channel-\1", text)
    text = CUSTOM_EMOJI_RE.sub(r":\1:", text)
    text = WHITESPACE_RE.sub(" ", text)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def estimate_token_count(text: str) -> int:
    if not text:
        return 0
    if tiktoken is not None:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    return max(1, len(text) // 4)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def shorten(text: str, max_chars: int = 280) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def display_text(text: str, max_chars: int = 900) -> str:
    normalized = _normalize_for_display(text)
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def display_excerpt(text: str, max_chars: int = 700) -> str:
    return display_text(_strip_chunk_heading(text), max_chars=max_chars)


def display_quote(text: str, max_chars: int = 700) -> str:
    excerpt = display_excerpt(text, max_chars=max_chars)
    lines = excerpt.splitlines() or [""]
    return "\n".join(f"> {line}" if line else ">" for line in lines)


def parse_iso_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _normalize_for_display(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = WHITESPACE_RE.sub(" ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"(?<!\n)(\s+)([-*]\s+)", r"\n\2", text)
    text = re.sub(r"(?<!\n)(\s+)(\d+\.\s+)", r"\n\2", text)
    text = re.sub(r"\s*;\s*", ";\n", text)
    text = re.sub(r"\s*//\s*", " // ", text)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def _strip_chunk_heading(text: str) -> str:
    lines = text.splitlines()
    while lines and (
        lines[0].startswith("Category: ")
        or lines[0].startswith("Channel: #")
        or not lines[0].strip()
    ):
        lines.pop(0)
    return "\n".join(lines).strip()
