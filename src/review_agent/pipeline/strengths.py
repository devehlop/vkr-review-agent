"""Сильные стороны товара и рекомендации для карточки (FR-13, вторая часть)."""
from dataclasses import dataclass, field

from pydantic import BaseModel

from review_agent.config import load_prompt
from review_agent.llm.base import LLMError
from review_agent.llm.gateway import FormatError, LLMGateway
from review_agent.models import Review, Sentence
from review_agent.pipeline.aggregate import AspectStats
from review_agent.pipeline.extract import Mention, _norm
from review_agent.pipeline.recommend import Quote, percent

PROMPT_VERSION = "strength_v1"


class TipDraft(BaseModel):
    action: str
    rationale: str = ""
    quotes: list[Quote]


@dataclass
class CardTip:
    """Что подчеркнуть в карточке товара."""

    attribute: str
    coverage: float
    positive_share: float
    action: str
    rationale: str
    quotes: list[tuple[int, str]] = field(default_factory=list)


def find_strengths(stats: list[AspectStats], own_product: str, min_mentions: int = 5,
                   min_positive: float = 0.7) -> list[AspectStats]:
    """Сильная сторона: атрибут, который упоминают достаточно часто и в основном положительно."""
    strong = [s for s in stats if s.product == own_product and not s.feature
              and s.mentions >= min_mentions and s.positive / s.mentions >= min_positive]
    return sorted(strong, key=lambda s: -s.coverage)


def card_tips(gateway: LLMGateway, stats: list[AspectStats], reviews: list[Review], sentences: list[Sentence],
              mentions: dict[int, list[Mention]], own_product: str, category: str = "одежда",
              limit: int = 3, examples: int = 8) -> list[CardTip]:
    """По самым заметным сильным сторонам просит модель предложить правку карточки товара.

    Цитаты проверяются так же, как у рекомендаций по проблемам: дословно из показанных предложений.
    """
    product_by_review = {r.id: r.product for r in reviews}
    tips = []
    for item in find_strengths(stats, own_product)[:limit]:
        shown = [s for s in sentences if product_by_review.get(s.review_id) == own_product
                 and any(m.attribute == item.attribute and m.sentiment == 1 for m in mentions.get(s.id, []))][:examples]
        if not shown:
            continue
        texts = {s.id: _norm(s.text) for s in shown}
        positive = item.positive / item.mentions

        def check(draft: TipDraft) -> list[str]:
            problems = [] if draft.action.strip() else ["пустое действие"]
            if not 1 <= len(draft.quotes) <= 3:
                problems.append("цитат должно быть от 1 до 3")
            for quote in draft.quotes:
                if quote.sentence_id not in texts:
                    problems.append(f"предложения {quote.sentence_id} нет в списке")
                elif not _norm(quote.quote) or _norm(quote.quote) not in texts[quote.sentence_id]:
                    problems.append(f"цитаты «{quote.quote}» нет в предложении {quote.sentence_id}")
            return problems

        prompt = load_prompt(PROMPT_VERSION).format(
            category=category, attribute=item.attribute, coverage=percent(item.coverage), positive=percent(positive),
            sentences="\n".join(f"{s.id}. {s.text}" for s in shown))
        try:
            draft = gateway.ask_json("strength", [{"role": "user", "content": prompt}], TipDraft, check=check, max_tokens=600)
        except (FormatError, LLMError):
            continue  # сильная сторона останется в отчёте без совета
        tips.append(CardTip(item.attribute, item.coverage, round(positive, 4), draft.action.strip(),
                            draft.rationale.strip(), [(q.sentence_id, q.quote) for q in draft.quotes]))
    return tips
