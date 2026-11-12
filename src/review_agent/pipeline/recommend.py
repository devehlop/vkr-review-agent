"""Шаг 8 конвейера: генерация рекомендаций по проблеме (FR-13, FR-14)."""
from typing import Literal

from pydantic import BaseModel

from review_agent.config import load_prompt
from review_agent.llm.gateway import LLMGateway
from review_agent.models import Issue, Sentence
from review_agent.pipeline.compare import NO_DATA, Comparison
from review_agent.pipeline.extract import _norm

PROMPT_VERSION = "recommend_v1"
TYPES = ("товар", "упаковка", "карточка товара", "размерная сетка", "логистика")


class Quote(BaseModel):
    sentence_id: int
    quote: str


class Draft(BaseModel):
    """Рекомендация в том виде, в каком её возвращает модель."""

    action: str
    type: str
    rationale: str
    expected_effect: str = ""
    effort: Literal[1, 2, 3]
    quotes: list[Quote]


class DraftBatch(BaseModel):
    recommendations: list[Draft]


def percent(share: float) -> str:
    return f"{share * 100:.0f} %"


def issue_context(issue: Issue, sentences: list[Sentence], comparison: Comparison | None,
                  category: str, limit: int = 8) -> dict:
    """Сведения о проблеме для промптов генератора и критика."""
    by_id = {s.id: s for s in sentences}
    shown = [by_id[i] for i in issue.sentence_ids if i in by_id][:limit]
    if comparison is None or comparison.verdict == NO_DATA or not comparison.competitors:
        competitors = "данных о конкурентах нет"
    else:
        parts = [f"{name}: доля негатива {percent(neg)}" for name, (neg, _) in comparison.competitors.items()]
        competitors = f"у нашего товара доля негатива {percent(comparison.own_neg_share)}; " + "; ".join(parts) \
                      + f". Наш товар {comparison.verdict}, чем конкуренты."
    return {
        "category": category,
        "title": issue.title,
        "attribute": issue.attribute,
        "coverage": percent(issue.coverage),
        "neg_share": percent(issue.neg_share),
        "competitors": competitors,
        "sentences": "\n".join(f"{s.id}. {s.text}" for s in shown),
        "shown": shown,
    }


def check_drafts(batch: DraftBatch, shown: list[Sentence], max_count: int) -> list[str]:
    """Проверка ответа генератора: тип из списка, цитаты дословно из показанных предложений."""
    problems = []
    texts = {s.id: _norm(s.text) for s in shown}
    if not 1 <= len(batch.recommendations) <= max_count:
        problems.append(f"рекомендаций должно быть от 1 до {max_count}, получено {len(batch.recommendations)}")
    for number, draft in enumerate(batch.recommendations, start=1):
        if not draft.action.strip():
            problems.append(f"рекомендация {number}: пустое действие")
        if draft.type not in TYPES:
            problems.append(f"рекомендация {number}: тип «{draft.type}» должен быть одним из: {', '.join(TYPES)}")
        if not 1 <= len(draft.quotes) <= 3:
            problems.append(f"рекомендация {number}: цитат должно быть от 1 до 3")
        for quote in draft.quotes:
            if quote.sentence_id not in texts:
                problems.append(f"рекомендация {number}: предложения {quote.sentence_id} нет в списке")
            elif not _norm(quote.quote) or _norm(quote.quote) not in texts[quote.sentence_id]:
                problems.append(f"рекомендация {number}: цитаты «{quote.quote}» нет в предложении {quote.sentence_id}")
    return problems


def generate(gateway: LLMGateway, context: dict, count: int = 2, revision: str = "") -> list[Draft]:
    """Запрашивает у модели рекомендации по проблеме.

    revision: замечания критика к прошлой версии. Тогда запрашивается одна исправленная рекомендация.
    """
    if revision:
        count = 1
    prompt = load_prompt(PROMPT_VERSION).format(
        **{k: v for k, v in context.items() if k != "shown"},
        revision=f"\n{revision}\n" if revision else "",
        count="одно действие" if count == 1 else f"от 1 до {count} действий",
        types=", ".join(TYPES),
    )
    batch = gateway.ask_json(
        "recommend", [{"role": "user", "content": prompt}], DraftBatch,
        check=lambda b: check_drafts(b, context["shown"], count), max_tokens=1500)
    return batch.recommendations
