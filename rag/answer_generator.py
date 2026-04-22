from __future__ import annotations

import re
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

from openai import OpenAI

from db.models import SourceReference
from utils.config import Settings
from utils.text import shorten


REFUSAL_MESSAGE = "This information is not found in your Discord notes."


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    refused: bool


@dataclass(frozen=True)
class StudyEvaluation:
    feedback: str


@dataclass(frozen=True)
class AnswerContext:
    chunk_id: str
    text: str
    similarity: float


@dataclass(frozen=True)
class SourceSelectionCandidate:
    source_id: str
    label: str
    snippet: str
    similarity: float


class AnswerGenerator(ABC):
    @abstractmethod
    def generate(self, question: str, contexts: list[AnswerContext]) -> GeneratedAnswer:
        raise NotImplementedError

    def choose_competing_sources(
        self,
        question: str,
        candidates: list[SourceSelectionCandidate],
    ) -> list[str]:
        return []

    def generate_study_question(self, excerpt: str, source_label: str) -> str:
        return f"What information is listed in this source note from {source_label}?"

    def evaluate_study_answer(self, question: str, source_answer: str, user_answer: str) -> StudyEvaluation:
        source_terms = _important_terms(source_answer)
        user_terms = set(_important_terms(user_answer))
        matched = [term for term in source_terms if term in user_terms]
        if not source_terms:
            verdict = "I could not extract enough terms from the source answer to compare reliably."
        elif len(matched) / len(source_terms) >= 0.6:
            verdict = "Good match. Your answer covered most of the key source terms."
        elif matched:
            verdict = "Partial match. You covered some source terms, but missed important details."
        else:
            verdict = "Weak match. Your answer did not overlap much with the source answer."
        missing = [term for term in source_terms if term not in user_terms][:8]
        missing_text = ", ".join(missing) if missing else "none"
        return StudyEvaluation(
            feedback=f"{verdict}\nMissing/unclear source terms: {missing_text}"
        )


class OpenAIAnswerGenerator(AnswerGenerator):
    def __init__(self, api_key: str, model: str, project_id: str | None = None) -> None:
        self.client = OpenAI(api_key=api_key, project=project_id)
        self.model = model

    def generate(self, question: str, contexts: list[AnswerContext]) -> GeneratedAnswer:
        if not contexts:
            return GeneratedAnswer(answer=REFUSAL_MESSAGE, refused=True)

        context_text = "\n\n".join(
            f"SOURCE {index + 1} ({context.chunk_id}, similarity={context.similarity:.3f})\n"
            f"{context.text}"
            for index, context in enumerate(contexts)
        )

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a local Discord study-notes assistant. Answer only "
                        "from the provided Discord note sources. Do not use outside "
                        "knowledge. Do not guess. Do not debug the user's code, infer "
                        "causes, generate fixes, write new code, or prescribe how "
                        "something must be done unless the provided notes explicitly "
                        "say that. If the user asks why something does not work, what "
                        "is wrong, or how to fix it, only report the note-backed facts, "
                        "rules, definitions, examples, or patterns that could help "
                        "them reason about it. If the notes do not contain enough "
                        f"evidence, reply exactly: {REFUSAL_MESSAGE}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\n"
                        f"Discord note sources:\n{context_text}\n\n"
                        "Write a study-focused answer grounded only in the sources. "
                        "Phrase uncertain or diagnostic requests as 'The notes say...' "
                        "or 'Relevant notes...' rather than as a fix. Use short bullet "
                        "points for lists or table items. Do not add examples, causes, "
                        "steps, recommendations, or rewritten code unless they appear "
                        "in the provided notes. Do not include source citations in the "
                        "answer body because the application adds sources separately."
                    ),
                },
            ],
        )
        answer = (response.choices[0].message.content or "").strip()
        refused = REFUSAL_MESSAGE in answer
        return GeneratedAnswer(answer=answer or REFUSAL_MESSAGE, refused=refused or not answer)

    def generate_study_question(self, excerpt: str, source_label: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Create one clear revision question using only the provided "
                        "source excerpt. Do not add facts. The question must match "
                        "the expected answer exactly. Do not narrow a full list into "
                        "'top three', 'main two', or similar unless the source excerpt "
                        "itself explicitly uses that limit. If the excerpt contains a "
                        "list, ask for the listed items, not an arbitrary subset."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Source label: {source_label}\n\n"
                        f"Source excerpt:\n{shorten(excerpt, 1200)}\n\n"
                        "Write one concise study question. Do not include the answer. "
                        "Make sure a correct answer would cover the same scope as the "
                        "source excerpt."
                    ),
                },
            ],
        )
        question = (response.choices[0].message.content or "").strip()
        return question or super().generate_study_question(excerpt, source_label)

    def choose_competing_sources(
        self,
        question: str,
        candidates: list[SourceSelectionCandidate],
    ) -> list[str]:
        if len(candidates) < 2:
            return []

        candidate_text = "\n\n".join(
            f"SOURCE_ID: {candidate.source_id}\n"
            f"LABEL: {candidate.label}\n"
            f"SIMILARITY: {candidate.similarity:.3f}\n"
            f"EXCERPT:\n{shorten(candidate.snippet, 650)}"
            for candidate in candidates[:8]
        )
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You decide whether a Discord notes question needs the user "
                        "to choose between multiple sources. Use only the candidate "
                        "labels and excerpts. Return JSON only. Return multiple "
                        "source_ids only when two or more sources are genuinely "
                        "plausible targets for the user's exact request and answering "
                        "from all of them would likely mix separate topics/classes. "
                        "If the request can be answered from the combined retrieved "
                        "notes, or one source is clearly best, return an empty list."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\n"
                        f"Candidate sources:\n{candidate_text}\n\n"
                        'Return JSON in this exact shape: {"source_ids": ["id1", "id2"]}'
                    ),
                },
            ],
        )
        raw = (response.choices[0].message.content or "").strip()
        selected = _parse_source_ids(raw)
        valid_ids = {candidate.source_id for candidate in candidates}
        return [source_id for source_id in selected if source_id in valid_ids]

    def evaluate_study_answer(self, question: str, source_answer: str, user_answer: str) -> StudyEvaluation:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Compare the student's answer to the source answer only, but "
                        "grade only the information requested by the question. Do not "
                        "penalize the student for omitting source details that the "
                        "question did not ask for. Do not use outside knowledge. Be "
                        "concise and practical. Say whether it is a good, partial, or "
                        "weak match, then list the main missing or incorrect points "
                        "needed to answer the question."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\n"
                        f"Source answer:\n{shorten(source_answer, 1600)}\n\n"
                        f"Student answer:\n{shorten(user_answer, 1000)}"
                    ),
                },
            ],
        )
        feedback = (response.choices[0].message.content or "").strip()
        return StudyEvaluation(feedback=feedback or "No feedback was generated.")


class ExtractiveAnswerGenerator(AnswerGenerator):
    """Conservative local provider that returns relevant snippets only."""

    def generate(self, question: str, contexts: list[AnswerContext]) -> GeneratedAnswer:
        if not contexts:
            return GeneratedAnswer(answer=REFUSAL_MESSAGE, refused=True)
        snippets = "\n".join(
            f"- {shorten(context.text, 350)}"
            for context in contexts[:3]
        )
        return GeneratedAnswer(
            answer=f"I found these relevant Discord note excerpts:\n{snippets}",
            refused=False,
        )

    def choose_competing_sources(
        self,
        question: str,
        candidates: list[SourceSelectionCandidate],
    ) -> list[str]:
        terms = set(_important_terms(question))
        if not terms:
            return []
        selected = [
            candidate.source_id
            for candidate in candidates
            if len(terms.intersection(_important_terms(candidate.snippet))) >= 2
        ]
        return selected if len(selected) >= 2 else []


STOPWORDS = {
    "about", "after", "again", "also", "and", "are", "because", "been", "but",
    "can", "could", "does", "for", "from", "has", "have", "into", "its", "not",
    "one", "only", "that", "the", "their", "then", "there", "these", "this",
    "was", "were", "what", "when", "where", "which", "with", "your",
}


def _important_terms(text: str) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]{2,}", text.lower()):
        if term in STOPWORDS or term in seen:
            continue
        seen.add(term)
        terms.append(term)
        if len(terms) >= 20:
            break
    return terms


def _parse_source_ids(raw: str) -> list[str]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match is None:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    source_ids = data.get("source_ids") if isinstance(data, dict) else None
    if not isinstance(source_ids, list):
        return []
    return [str(source_id) for source_id in source_ids]


def create_answer_generator(settings: Settings) -> AnswerGenerator:
    if settings.answer_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI answer generation.")
        return OpenAIAnswerGenerator(
            api_key=settings.openai_api_key,
            model=settings.openai_answer_model,
            project_id=settings.openai_project_id,
        )
    if settings.answer_provider == "extractive":
        return ExtractiveAnswerGenerator()
    raise RuntimeError(f"Unsupported ANSWER_PROVIDER: {settings.answer_provider}")
