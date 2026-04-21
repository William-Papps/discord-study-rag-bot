from __future__ import annotations

import random
import re
from dataclasses import dataclass

from db.models import ChunkRecord, NoteSource, SourceReference
from db.repositories import ChunkRepository, MessageRepository
from rag.answer_generator import AnswerGenerator
from utils.text import shorten


@dataclass(frozen=True)
class FindResult:
    channel_name: str
    category_name: str | None
    timestamp: str
    message_url: str
    snippet: str


@dataclass(frozen=True)
class StudyCard:
    question: str
    answer: str
    source: SourceReference | None


@dataclass(frozen=True)
class StudyScope:
    label: str
    channel_ids: list[int]
    source_count: int


class StudyService:
    def __init__(
        self,
        message_repository: MessageRepository,
        chunk_repository: ChunkRepository,
        answer_generator: AnswerGenerator,
    ) -> None:
        self.message_repository = message_repository
        self.chunk_repository = chunk_repository
        self.answer_generator = answer_generator

    def find(
        self,
        query: str,
        scope: str | None = None,
        limit: int = 8,
    ) -> list[FindResult]:
        resolved = self.resolve_scope(scope, allow_all=True)
        if resolved.channel_ids:
            messages = self.message_repository.search_channels(
                query=query,
                channel_ids=resolved.channel_ids,
                limit=limit,
            )
        else:
            messages = self.message_repository.search(query=query, limit=limit)
        return [
            FindResult(
                channel_name=message.channel_name,
                category_name=message.category_name,
                timestamp=message.timestamp,
                message_url=message.message_url,
                snippet=shorten(message.cleaned_content, 300),
            )
            for message in messages
        ]

    def summarize_scope(self, scope: str, limit: int = 12) -> tuple[StudyScope, list[StudyCard]]:
        resolved = self.resolve_scope(scope, allow_all=False)
        chunks = self.chunk_repository.list_for_channels(channel_ids=resolved.channel_ids)
        if not chunks:
            return resolved, []

        selected = _spread(chunks, limit)
        return resolved, self._cards_from_chunks(
            chunks=selected,
            topic="",
            question_prefix="Review this exact note excerpt",
        )

    def make_quiz(
        self,
        scope: str,
        topic: str = "",
        count: int = 5,
    ) -> tuple[StudyScope, list[StudyCard]]:
        resolved = self.resolve_scope(scope, allow_all=False)
        chunks = self._study_chunks(
            channel_ids=resolved.channel_ids,
            topic=topic,
            limit=count,
        )
        return resolved, self._cards_from_chunks(
            chunks=chunks,
            topic=topic,
            question_prefix="Answer from memory before checking the source",
        )

    def make_flashcards(
        self,
        scope: str,
        topic: str = "",
        count: int = 5,
    ) -> tuple[StudyScope, list[StudyCard]]:
        resolved = self.resolve_scope(scope, allow_all=False)
        chunks = self._study_chunks(
            channel_ids=resolved.channel_ids,
            topic=topic,
            limit=count,
        )
        return resolved, self._cards_from_chunks(
            chunks=chunks,
            topic=topic,
            question_prefix="What does the source note say",
        )

    def resolve_scope(self, raw_scope: str | None, allow_all: bool) -> StudyScope:
        sources = self.message_repository.list_sources()
        scope = (raw_scope or "").strip()
        if not scope:
            if allow_all:
                return StudyScope(
                    label="all synced notes",
                    channel_ids=[source.channel_id for source in sources],
                    source_count=len(sources),
                )
            raise ValueError("Scope is required. Use a category like DES502 or a synced thread/forum name.")

        normalized = _normalize(scope)
        if scope.isdigit():
            matching = [source for source in sources if source.channel_id == int(scope)]
            if matching:
                return _source_scope(matching[0])

        category_matches = [
            source for source in sources
            if source.category_name and _normalize(source.category_name) == normalized
        ]
        if category_matches:
            return _category_scope(category_matches, category_matches[0].category_name or scope)

        source_matches = [
            source for source in sources
            if _normalize(source.channel_name) == normalized
        ]
        if source_matches:
            return _source_scope(source_matches[0])

        partial_categories = {
            source.category_name
            for source in sources
            if source.category_name and normalized in _normalize(source.category_name)
        }
        if len(partial_categories) == 1:
            category_name = next(iter(partial_categories))
            return _category_scope(
                [source for source in sources if source.category_name == category_name],
                category_name or scope,
            )

        partial_sources = [
            source for source in sources
            if normalized in _normalize(source.channel_name)
        ]
        if len(partial_sources) == 1:
            return _source_scope(partial_sources[0])

        suggestions = _scope_suggestions(sources)
        if partial_categories or partial_sources:
            raise ValueError(
                "That scope matched more than one source. Be more specific. "
                f"Examples: {suggestions}"
            )
        raise ValueError(f"No synced source matched {scope!r}. Examples: {suggestions}")

    def _study_chunks(
        self,
        channel_ids: list[int],
        topic: str,
        limit: int,
    ) -> list[ChunkRecord]:
        all_chunks = self.chunk_repository.list_for_channels(channel_ids=channel_ids)
        if topic.strip():
            chunks = _matching_chunks(
                chunks=all_chunks,
                topic=topic,
            )
            if chunks:
                return _sample_chunks(chunks, limit)
        return _sample_chunks(all_chunks, limit)

    def _cards_from_chunks(
        self,
        chunks: list[ChunkRecord],
        topic: str,
        question_prefix: str,
    ) -> list[StudyCard]:
        references = {
            reference.chunk_id: reference
            for reference in self.chunk_repository.get_source_references(
                [chunk.chunk_id for chunk in chunks]
            )
        }
        cards: list[StudyCard] = []
        for chunk in chunks:
            excerpt = _best_excerpt(chunk.chunk_text, topic)
            if not excerpt:
                continue
            question = _make_question(
                answer_generator=self.answer_generator,
                prefix=question_prefix,
                topic=topic,
                excerpt=excerpt,
                source=references.get(chunk.chunk_id),
            )
            cards.append(
                StudyCard(
                    question=question,
                    answer=excerpt,
                    source=references.get(chunk.chunk_id),
                )
            )
        return cards

    def evaluate_flashcard_answer(self, card: StudyCard, user_answer: str) -> str:
        evaluation = self.answer_generator.evaluate_study_answer(
            question=card.question,
            source_answer=card.answer,
            user_answer=user_answer,
        )
        return evaluation.feedback


def format_find_results(results: list[FindResult]) -> str:
    if not results:
        return "No matching synced notes were found."

    lines = ["**Find results**"]
    for index, result in enumerate(results, start=1):
        source_name = (
            f"{result.category_name} / #{result.channel_name}"
            if result.category_name
            else f"#{result.channel_name}"
        )
        lines.append(
            f"{index}. {source_name}\n"
            f"   {result.timestamp}\n"
            f"   {result.message_url}\n"
            f"   `{result.snippet}`"
        )
    return "\n".join(lines)


def format_quiz(cards: list[StudyCard]) -> str:
    if not cards:
        return "No synced notes were found for that quiz request."

    lines = ["**Quiz**", "Answer from memory, then check the hidden source excerpts."]
    for index, card in enumerate(cards, start=1):
        lines.append(f"{index}. {card.question}")
    lines.append("")
    lines.append("**Answer key**")
    for index, card in enumerate(cards, start=1):
        source = _format_source(card.source)
        lines.append(f"{index}. ||{shorten(card.answer, 500)}||{source}")
    return "\n".join(lines)


def format_summary(cards: list[StudyCard], scope_label: str) -> str:
    if not cards:
        return f"No synced chunks were found for {scope_label}."

    lines = [
        f"**Extractive summary for {scope_label}**",
        "These are direct note excerpts, not rewritten summaries.",
    ]
    for index, card in enumerate(cards, start=1):
        source = _format_source(card.source)
        lines.append(f"{index}. {shorten(card.answer, 500)}{source}")
    return "\n".join(lines)


def format_flashcard(card: StudyCard, index: int, total: int) -> str:
    return f"**Flashcard {index}/{total}**\n{card.question}"


def format_flashcard_answer(card: StudyCard, user_answer: str, feedback: str) -> str:
    source = _format_source(card.source)
    return (
        "**Your answer**\n"
        f"{shorten(user_answer, 700)}\n\n"
        "**Comparison**\n"
        f"{shorten(feedback, 900)}\n\n"
        "**Source answer**\n"
        f"{shorten(card.answer, 900)}{source}"
    )


def _best_excerpt(text: str, topic: str) -> str:
    excerpts = _extract_excerpts(text)
    if not excerpts:
        return ""
    terms = [term.lower() for term in topic.split() if len(term) >= 2]
    if terms:
        matching = []
        for excerpt in excerpts:
            lowered = excerpt.lower()
            if any(term in lowered for term in terms):
                matching.append(excerpt)
        if matching:
            return shorten(random.choice(matching), 900)
    return shorten(random.choice(excerpts), 900)


def _extract_excerpts(text: str) -> list[str]:
    cleaned = re.sub(r"^(Category: [^\n]+\n)?Channel: #[^\n]+\n*", "", text.strip())
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", cleaned) if part.strip()]
    excerpts: list[str] = []
    for paragraph in paragraphs:
        paragraph = re.sub(r"^\[[^\]]+\]\s+[^:]{1,100}:\s*", "", paragraph)
        paragraph = " ".join(paragraph.split())
        if len(paragraph) >= 25:
            excerpts.append(paragraph)
    return excerpts


def _make_question(
    answer_generator: AnswerGenerator,
    prefix: str,
    topic: str,
    excerpt: str,
    source: SourceReference | None,
) -> str:
    topic = topic.strip()
    source_name = _source_label(source)
    if topic:
        generated = answer_generator.generate_study_question(excerpt, source_name)
        return f"{generated}\nTopic focus: {topic}"
    return answer_generator.generate_study_question(excerpt, source_name)


def _format_source(source: SourceReference | None) -> str:
    if source is None:
        return ""
    return f"\n   Source: {_source_label(source)} {source.timestamp}\n   {source.message_url}"


def _source_label(source: SourceReference | None) -> str:
    if source is None:
        return "the selected scope"
    category = f"{source.category_name} / " if source.category_name else ""
    return f"{category}#{source.channel_name}"


def _source_scope(source: NoteSource) -> StudyScope:
    category = f"{source.category_name} / " if source.category_name else ""
    return StudyScope(
        label=f"{category}{source.channel_name}",
        channel_ids=[source.channel_id],
        source_count=1,
    )


def _category_scope(sources: list[NoteSource], category_name: str) -> StudyScope:
    return StudyScope(
        label=f"category {category_name}",
        channel_ids=[source.channel_id for source in sources],
        source_count=len(sources),
    )


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _scope_suggestions(sources: list[NoteSource]) -> str:
    categories = []
    for source in sources:
        if source.category_name and source.category_name not in categories:
            categories.append(source.category_name)
    examples = categories[:3] or [source.channel_name for source in sources[:3]]
    return ", ".join(examples) if examples else "sync notes first"


def _spread(chunks: list[ChunkRecord], limit: int) -> list[ChunkRecord]:
    if len(chunks) <= limit:
        return chunks
    if limit <= 1:
        return [chunks[0]]
    step = (len(chunks) - 1) / (limit - 1)
    indexes = [round(index * step) for index in range(limit)]
    deduped = list(dict.fromkeys(indexes))
    return [chunks[index] for index in deduped]


def _matching_chunks(chunks: list[ChunkRecord], topic: str) -> list[ChunkRecord]:
    terms = [_normalize(term) for term in topic.split() if len(term) >= 2]
    if not terms:
        return chunks
    matches = []
    for chunk in chunks:
        normalized = _normalize(chunk.chunk_text)
        if any(term and term in normalized for term in terms):
            matches.append(chunk)
    return matches


def _sample_chunks(chunks: list[ChunkRecord], limit: int) -> list[ChunkRecord]:
    if len(chunks) <= limit:
        shuffled = list(chunks)
        random.shuffle(shuffled)
        return shuffled
    return random.sample(chunks, limit)
