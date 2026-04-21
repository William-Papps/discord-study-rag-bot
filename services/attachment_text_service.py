from __future__ import annotations

import asyncio
import base64
import logging
from abc import ABC, abstractmethod

import discord
from openai import OpenAI, OpenAIError

from utils.config import Settings

logger = logging.getLogger(__name__)


IMAGE_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
}


class AttachmentTextService(ABC):
    @abstractmethod
    async def extract_text(self, attachment: discord.Attachment) -> str | None:
        raise NotImplementedError


class NoopAttachmentTextService(AttachmentTextService):
    async def extract_text(self, attachment: discord.Attachment) -> str | None:
        return None


class OpenAIAttachmentTextService(AttachmentTextService):
    def __init__(self, api_key: str, model: str, max_attachment_bytes: int) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_attachment_bytes = max_attachment_bytes
        self._disabled_reason: str | None = None

    async def extract_text(self, attachment: discord.Attachment) -> str | None:
        if self._disabled_reason:
            return None
        if not self._is_supported_image(attachment):
            return None
        if attachment.size and attachment.size > self.max_attachment_bytes:
            logger.info(
                "Skipping OCR for %s because size %s exceeds limit %s",
                attachment.filename,
                attachment.size,
                self.max_attachment_bytes,
            )
            return None

        try:
            image_bytes = await attachment.read()
            if len(image_bytes) > self.max_attachment_bytes:
                logger.info(
                    "Skipping OCR for %s because downloaded size exceeds limit",
                    attachment.filename,
                )
                return None
            return await asyncio.to_thread(self._extract_text_sync, attachment, image_bytes)
        except OpenAIError as exc:
            self._disabled_reason = (
                f"OpenAI OCR failed with model {self.model!r}: {exc}. "
                "Disabling image OCR for the rest of this run."
            )
            logger.error(self._disabled_reason)
            return None
        except Exception:
            logger.exception("Failed OCR for attachment %s", attachment.filename)
            return None

    def _extract_text_sync(self, attachment: discord.Attachment, image_bytes: bytes) -> str | None:
        content_type = attachment.content_type or "image/png"
        data_url = f"data:{content_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract only visible text from the provided Discord note image. "
                        "Preserve table-like structure where practical. If there is no "
                        "readable text, respond exactly: NO_READABLE_TEXT"
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Transcribe the visible text from this class note image.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url, "detail": "high"},
                        },
                    ],
                },
            ],
        )
        text = (response.choices[0].message.content or "").strip()
        if not text or text == "NO_READABLE_TEXT":
            return None
        return text

    @staticmethod
    def _is_supported_image(attachment: discord.Attachment) -> bool:
        content_type = attachment.content_type
        if content_type in IMAGE_CONTENT_TYPES:
            return True
        filename = attachment.filename.lower()
        return filename.endswith((".jpg", ".jpeg", ".png", ".webp", ".gif"))


def create_attachment_text_service(settings: Settings) -> AttachmentTextService:
    if settings.image_text_provider in {"none", "off", "disabled"}:
        return NoopAttachmentTextService()
    if settings.image_text_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for image text extraction.")
        return OpenAIAttachmentTextService(
            api_key=settings.openai_api_key,
            model=settings.openai_vision_model,
            max_attachment_bytes=settings.max_attachment_bytes,
        )
    raise RuntimeError(f"Unsupported IMAGE_TEXT_PROVIDER: {settings.image_text_provider}")
