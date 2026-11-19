"""Шаг 7. Сравнение с конкурентами по атрибутам (FR-12).

Сравнивается доля отзывов с негативом по атрибуту (охват × доля негатива):
это понятная продавцу величина «сколько покупателей жалуются на X».
"""

from __future__ import annotations

from dataclasses import dataclass

from review_agent.config import Thresholds
from review_agent.models import AspectStats, Issue
from review_agent.pipeline.prioritize import competitor_is_better, negative_rate


@dataclass
class ComparisonRow:
    attribute: str
    own_rate: float
    competitor_rate: float | None
    # «хуже» — у своего товара жалоб заметно больше, «лучше» — заметно меньше
    verdict: str


def compare_attributes(
    stats: list[AspectStats],
    own_ids: set[str],
    ratio: float = Thresholds.competitor_ratio,
) -> list[ComparisonRow]:
    own = {s.attribute: s for s in stats if s.product_id in own_ids and s.feature is None}
    comp: dict[str, list[AspectStats]] = {}
    for s in stats:
        if s.product_id not in own_ids and s.feature is None:
            comp.setdefault(s.attribute, []).append(s)
    competitor_ids = {s.product_id for s in stats if s.product_id not in own_ids}

    rows = []
    for attr in sorted(set(own) | set(comp)):
        own_rate = negative_rate(own[attr]) if attr in own else 0.0
        if not competitor_ids:
            rows.append(ComparisonRow(attr, own_rate, None, "нет данных"))
            continue
        # Конкурент без упоминаний атрибута — ноль жалоб, а не отсутствие данных
        rates = [negative_rate(s) for s in comp.get(attr, [])]
        rates += [0.0] * (len(competitor_ids) - len(rates))
        comp_rate = sum(rates) / len(rates)
        if own_rate > 0 and comp_rate <= own_rate * ratio:
            verdict = "хуже"
        elif comp_rate > 0 and own_rate <= comp_rate * ratio:
            verdict = "лучше"
        else:
            verdict = "на уровне"
        rows.append(ComparisonRow(attr, own_rate, comp_rate, verdict))
    return sorted(rows, key=lambda r: -(r.own_rate - (r.competitor_rate or 0)))


def competitor_flags(issues: list[Issue], stats: list[AspectStats], own_ids: set[str]) -> dict[str, bool]:
    """Для каждой проблемы: у конкурентов этот атрибут заметно лучше (K_конк в формуле)."""
    own = {(s.product_id, s.attribute): s for s in stats if s.product_id in own_ids and s.feature is None}
    flags = {}
    for issue in issues:
        comps = [s for s in stats if s.product_id not in own_ids and s.feature is None and s.attribute == issue.attribute]
        flags[issue.id] = competitor_is_better(own.get((issue.product_id, issue.attribute)), comps)
    return flags
