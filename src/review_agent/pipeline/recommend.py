"""Шаги 8–9. Генерация рекомендаций и агент-критик (FR-13…FR-15).

Генератор предлагает 1–2 действия по проблеме. Критик ставит оценки по
четырём критериям; решение «прошла / не прошла» принимает код по порогу,
а не модель. Не прошедшая рекомендация возвращается генератору вместе
с замечаниями, не более max_rounds раз. После этого она в отчёт не
попадает, но остаётся в журнале проверок.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from review_agent.config import Thresholds
from review_agent.llm.gateway import LLMFormatError, LLMGateway
from review_agent.models import AspectStats, Issue, Mention, Recommendation, RecommendationType
from review_agent.pipeline.extract import quote_in_sentence
from review_agent.prompts import render_prompt

CRITERIA = ("specificity", "grounding", "feasibility", "novelty")
PASS_SCORE = 4


class DraftRecommendation(BaseModel):
    action: str = Field(min_length=10)
    type: RecommendationType
    rationale: str = Field(min_length=10)
    expected_effect: str = ""
    effort: int = Field(ge=1, le=3)
    quotes: list[str] = Field(min_length=1)


class DraftList(BaseModel):
    recommendations: list[DraftRecommendation] = Field(min_length=1, max_length=2)


class Scores(BaseModel):
    specificity: int = Field(ge=1, le=5)
    grounding: int = Field(ge=1, le=5)
    feasibility: int = Field(ge=1, le=5)
    novelty: int = Field(ge=1, le=5)


class CriticAnswer(BaseModel):
    scores: Scores
    comment: str = ""


@dataclass
class CritiqueRecord:
    recommendation_id: str
    iteration: int
    scores: dict[str, int]
    comment: str
    passed: bool


@dataclass
class RecommendResult:
    accepted: list[Recommendation] = field(default_factory=list)
    # Не прошли критика после всех доработок
    rejected: list[Recommendation] = field(default_factory=list)
    critiques: list[CritiqueRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class Target:
    """То, на что направлена рекомендация: проблема или сильная сторона."""

    key: str
    issue_id: str | None
    prompt: str
    values: dict[str, str]
    description: str
    quotes: list[str]


def _pct(x: float) -> str:
    return f"{x * 100:.0f} %"


def issue_target(issue: Issue, mentions: dict[str, Mention], competitor_better: bool, max_quotes: int = 7) -> Target:
    quotes: list[str] = []
    for mid in issue.mention_ids:
        q = mentions[mid].quote
        if q not in quotes:
            quotes.append(q)
    quotes = quotes[:max_quotes]
    competitor = (
        "у конкурентов жалобы на этот атрибут встречаются заметно реже — это отставание от рынка."
        if competitor_better
        else "данных о преимуществе конкурентов по этому атрибуту нет."
    )
    return Target(
        key=issue.id,
        issue_id=issue.id,
        prompt="recommend",
        values={
            "title": issue.title,
            "attribute": issue.attribute,
            "coverage": _pct(issue.coverage),
            "neg_share": _pct(issue.neg_share),
            "competitor": competitor,
        },
        description=f"проблема «{issue.title}» ({issue.attribute}), {_pct(issue.coverage)} отзывов",
        quotes=quotes,
    )


def strength_target(stats: AspectStats, quotes: list[str]) -> Target:
    return Target(
        key=f"S:{stats.attribute}",
        issue_id=None,
        prompt="strengths",
        values={"attribute": stats.attribute, "coverage": _pct(stats.coverage), "pos_share": _pct(stats.pos_share)},
        description=f"сильная сторона «{stats.attribute}», {_pct(stats.pos_share)} позитивных упоминаний",
        quotes=quotes,
    )


def select_strengths(
    stats: list[AspectStats],
    mentions: list[Mention],
    product_of_mention: dict[str, str],
    own_ids: set[str],
    min_pos_share: float = 0.6,
    min_mentions: int = Thresholds.min_support,
    limit: int = 3,
    max_quotes: int = 5,
) -> list[Target]:
    """Атрибуты своего товара, которые чаще хвалят, чем ругают."""
    candidates = [
        s for s in stats
        if s.product_id in own_ids and s.feature is None and s.attribute != "Общее впечатление"
        and s.pos_share >= min_pos_share and s.mentions >= min_mentions
    ]
    candidates.sort(key=lambda s: -s.coverage * s.pos_share)
    targets = []
    for s in candidates[:limit]:
        quotes: list[str] = []
        for m in mentions:
            if m.attribute == s.attribute and m.sentiment == 1 and product_of_mention[m.id] == s.product_id and m.quote not in quotes:
                quotes.append(m.quote)
        targets.append(strength_target(s, quotes[:max_quotes]))
    return targets


def _quotes_block(quotes: list[str]) -> str:
    return "\n".join(f"- {q}" for q in quotes)


def generate(gateway: LLMGateway, target: Target, feedback: str = "") -> list[DraftRecommendation]:
    values = dict(target.values, quotes=_quotes_block(target.quotes))
    values["feedback"] = f"\nЗамечания рецензента к прошлому варианту, учти их:\n{feedback}\n" if feedback else ""
    messages = render_prompt(target.prompt, "v1", **values)

    def validate(d: DraftList) -> list[str]:
        errors = []
        for k, r in enumerate(d.recommendations, start=1):
            for q in r.quotes:
                if not any(quote_in_sentence(q, src) or quote_in_sentence(src, q) for src in target.quotes):
                    errors.append(f"рекомендация {k}: цитаты «{q}» нет в списке, перепиши цитату точно")
        return errors

    return gateway.structured("recommend", messages, DraftList, validate).recommendations


def critique(gateway: LLMGateway, target: Target, draft: DraftRecommendation, others: list[str]) -> CriticAnswer:
    rec_text = json.dumps(draft.model_dump(mode="json"), ensure_ascii=False, indent=1)
    messages = render_prompt(
        "critic",
        "v1",
        problem=target.description,
        quotes=_quotes_block(target.quotes),
        recommendation=rec_text,
        others="\n".join(f"- {o}" for o in others) or "пока нет",
    )
    return gateway.structured("critic", messages, CriticAnswer)


def passed(scores: Scores, threshold: int = PASS_SCORE) -> bool:
    return all(getattr(scores, c) >= threshold for c in CRITERIA)


def recommend(
    gateway: LLMGateway,
    targets: list[Target],
    use_critic: bool = True,
    max_rounds: int = Thresholds.max_critic_rounds,
) -> RecommendResult:
    """Рекомендации для всех целей. use_critic=False — для сравнения вклада критика."""
    result = RecommendResult()
    for target in targets:
        feedback = ""
        counter = 0
        for round_no in range(max_rounds + 1):
            try:
                drafts = generate(gateway, target, feedback)
            except LLMFormatError as e:
                result.errors.append(str(e))
                break
            round_failed: list[tuple[DraftRecommendation, CriticAnswer]] = []
            for draft in drafts:
                counter += 1
                rec = Recommendation(
                    id=f"{target.key}-R{counter}",
                    issue_id=target.issue_id,
                    action=draft.action,
                    type=draft.type,
                    rationale=draft.rationale,
                    expected_effect=draft.expected_effect,
                    effort=draft.effort,
                    quotes=draft.quotes,
                )
                if not use_critic:
                    result.accepted.append(rec)
                    continue
                others = [r.action for r in result.accepted]
                try:
                    answer = critique(gateway, target, draft, others)
                except LLMFormatError as e:
                    result.errors.append(str(e))
                    continue
                ok = passed(answer.scores)
                result.critiques.append(CritiqueRecord(rec.id, round_no, answer.scores.model_dump(), answer.comment, ok))
                if ok:
                    result.accepted.append(rec)
                elif round_no == max_rounds:
                    result.rejected.append(rec)
                else:
                    round_failed.append((draft, answer))
            if not round_failed:
                break
            # На доработку отправляем только непрошедшие варианты вместе с замечаниями
            feedback = "\n".join(
                f"- «{d.action}»: оценки {a.scores.model_dump()}; {a.comment}" for d, a in round_failed
            )
    return result
