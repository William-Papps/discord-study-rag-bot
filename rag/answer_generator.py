from __future__ import annotations

import re
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


class AnswerGenerator(ABC):
    @abstractmethod
    def generate(self, question: str, contexts: list[AnswerContext]) -> GeneratedAnswer:
        raise NotImplementedError

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
    def __init__(self, api_key: str, model: str) -> None:
        self.client = OpenAI(api_key=api_key)
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
                        "You are a local Discord notes assistant. Answer only from the "
                        "provided Discord note sources. Do not use outside knowledge. "
                        "Do not guess. If the sources do not contain enough evidence, "
                        f"reply exactly: {REFUSAL_MESSAGE}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\n"
                        f"Discord note sources:\n{context_text}\n\n"
                        "Write a direct answer grounded only in the sources. "
                        "Use short bullet points for lists or table items. "
                        "Do not include source citations in the answer body because "
                        "the application adds sources separately."
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


def create_answer_generator(settings: Settings) -> AnswerGenerator:
    if settings.answer_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI answer generation.")
        return OpenAIAnswerGenerator(
            api_key=settings.openai_api_key,
            model=settings.openai_answer_model,
        )
    if settings.answer_provider == "extractive":
        return ExtractiveAnswerGenerator()
    raise RuntimeError(f"Unsupported ANSWER_PROVIDER: {settings.answer_provider}")
