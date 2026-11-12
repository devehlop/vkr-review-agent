"""Шаг 9 конвейера: цикл «генератор — критик» (FR-15).

Критик — отдельный запрос к модели, который оценивает рекомендацию по четырём критериям.
Рекомендация, не прошедшая проверку, возвращается генератору вместе с замечаниями.
"""
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from review_agent.config import load_prompt
from review_agent.llm.base import LLMError
from review_agent.llm.gateway import FormatError, LLMGateway
from review_agent.pipeline.recommend import Draft, generate

PROMPT_VERSION = "critic_v1"
CRITERIA = {"specificity": "конкретность", "grounding": "опора на отзывы",
            "feasibility": "выполнимость", "novelty": "отсутствие повтора"}


class Scores(BaseModel):
    specificity: int = Field(ge=1, le=5)
    grounding: int = Field(ge=1, le=5)
    feasibility: int = Field(ge=1, le=5)
    novelty: int = Field(ge=1, le=5)


class CriticReply(BaseModel):
    scores: Scores
    comment: str = ""


@dataclass
class Verdict:
    iteration: int
    scores: dict[str, int]
    comment: str
    passed: bool


@dataclass
class Reviewed:
    """Рекомендация после цикла проверки."""

    draft: Draft
    accepted: bool
    verdicts: list[Verdict] = field(default_factory=list)


def critique(gateway: LLMGateway, context: dict, draft: Draft, accepted_actions: list[str],
             iteration: int = 0, threshold: int = 4) -> Verdict:
    """Одна проверка рекомендации. Проходит, если все четыре оценки не ниже threshold."""
    prompt = load_prompt(PROMPT_VERSION).format(
        **{k: v for k, v in context.items() if k not in ("shown", "neg_share", "competitors")},
        action=draft.action, type=draft.type, rationale=draft.rationale,
        expected_effect=draft.expected_effect or "не указан", effort=draft.effort,
        accepted="\n".join(f"- {a}" for a in accepted_actions) or "пока нет",
        threshold=threshold,
    )
    reply = gateway.ask_json("critic", [{"role": "user", "content": prompt}], CriticReply, max_tokens=500)
    scores = reply.scores.model_dump()
    return Verdict(iteration, scores, reply.comment.strip(), all(v >= threshold for v in scores.values()))


def revision_text(draft: Draft, verdict: Verdict) -> str:
    scores = [f"{CRITERIA[name]}: {value} из 5" for name, value in verdict.scores.items()]
    return (
        "Предыдущая версия рекомендации отклонена экспертом.\n"
        f"Действие: {draft.action}\nОбоснование: {draft.rationale}\n"
        f"Оценки: {'; '.join(scores)}\nЗамечания эксперта: {verdict.comment or 'не указаны'}\n"
        "Предложи исправленную рекомендацию с учётом замечаний."
    )


def review_draft(gateway: LLMGateway, context: dict, draft: Draft, accepted_actions: list[str],
                 max_revisions: int = 2, threshold: int = 4) -> Reviewed:
    """Проверяет рекомендацию и при необходимости дорабатывает её не более max_revisions раз."""
    result = Reviewed(draft, accepted=False)
    for iteration in range(max_revisions + 1):
        try:
            verdict = critique(gateway, context, result.draft, accepted_actions, iteration, threshold)
        except (FormatError, LLMError) as error:
            # Без оценки критика рекомендацию в отчёт не пропускаем
            result.verdicts.append(Verdict(iteration, {}, f"критик не ответил: {error}", False))
            return result
        result.verdicts.append(verdict)
        if verdict.passed:
            result.accepted = True
            return result
        if iteration == max_revisions:
            return result
        try:
            result.draft = generate(gateway, context, revision=revision_text(result.draft, verdict))[0]
        except (FormatError, LLMError):
            return result
    return result
