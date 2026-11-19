"""Шаг 3. Справочник аспектов по выборке отзывов (FR-7).

Справочник один для своего товара и конкурентов, иначе их нельзя сравнить.
Продавец может поправить его в интерфейсе перед шагом 4.
"""

from __future__ import annotations

import random
from datetime import date

from pydantic import BaseModel, Field

from review_agent.llm.gateway import LLMGateway
from review_agent.models import Review
from review_agent.prompts import render_prompt
from review_agent.taxonomy import Taxonomy, TaxonomyAttribute

OVERALL = "Общее впечатление"


class DraftAttribute(BaseModel):
    attribute: str = Field(min_length=2)
    features: list[str] = Field(min_length=1)


class DraftTaxonomy(BaseModel):
    attributes: list[DraftAttribute]


def _validate(t: DraftTaxonomy) -> list[str]:
    errors = []
    names = [a.attribute.strip().lower() for a in t.attributes]
    if not 4 <= len(names) <= 15:
        errors.append(f"атрибутов должно быть 6–12, сейчас {len(names)}")
    dups = sorted({n for n in names if names.count(n) > 1})
    if dups:
        errors.append(f"повторяются атрибуты: {', '.join(dups)}")
    if OVERALL.lower() not in names:
        errors.append(f"нет атрибута «{OVERALL}»")
    return errors


def sample_reviews(reviews: list[Review], size: int = 150, seed: int = 2026) -> list[Review]:
    if len(reviews) <= size:
        return list(reviews)
    return random.Random(seed).sample(reviews, size)


def build_taxonomy(gateway: LLMGateway, reviews: list[Review], category: str, size: int = 150, max_chars: int = 400) -> Taxonomy:
    sample = sample_reviews(reviews, size)
    lines = []
    for r in sample:
        parts = [p for p in (f"Достоинства: {r.pros}" if r.pros else "", f"Недостатки: {r.cons}" if r.cons else "", r.text) if p]
        lines.append("- " + " ".join(parts)[:max_chars])
    messages = render_prompt("taxonomy", "v1", category=category, reviews="\n".join(lines))
    draft = gateway.structured("taxonomy", messages, DraftTaxonomy, _validate)
    attributes = []
    for k, a in enumerate(draft.attributes, start=1):
        features = list(dict.fromkeys(f.strip() for f in a.features if f.strip()))
        attributes.append(TaxonomyAttribute(code=f"a{k}", attribute=a.attribute.strip(), features=features))
    return Taxonomy(version=f"auto-{date.today().isoformat()}", description=f"Построен агентом для категории «{category}»", attributes=attributes)
