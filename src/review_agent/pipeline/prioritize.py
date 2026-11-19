"""Шаг 10. Приоритизация рекомендаций по объяснимой формуле (FR-16).

P = охват × доля негатива × K_конк / сложность

K_конк > 1, если у конкурентов эта проблема встречается заметно реже,
то есть это реальное отставание от рынка. Коэффициенты подбираются
на этапе тестирования (docs/architecture.md, шаг 10).
"""

from __future__ import annotations

from review_agent.config import Thresholds
from review_agent.models import AspectStats, Issue, Recommendation


def negative_rate(stats: AspectStats) -> float:
    """Доля отзывов с негативом по аспекту: охват × доля негатива."""
    return stats.coverage * stats.neg_share


def competitor_is_better(
    own: AspectStats | None,
    competitors: list[AspectStats],
    ratio: float = Thresholds.competitor_ratio,
) -> bool:
    """У конкурентов проблема заметно реже: средняя доля негатива не больше ratio от своей."""
    if own is None or not competitors or negative_rate(own) == 0:
        return False
    mean = sum(negative_rate(c) for c in competitors) / len(competitors)
    return mean <= negative_rate(own) * ratio


def priority(coverage: float, neg_share: float, effort: int, competitor_better: bool, k: float = Thresholds.competitor_k) -> float:
    if not 1 <= effort <= 3:
        raise ValueError("сложность должна быть от 1 до 3")
    return coverage * neg_share * (k if competitor_better else 1.0) / effort


def rank(
    recommendations: list[Recommendation],
    issues: list[Issue],
    competitor_better_by_issue: dict[str, bool] | None = None,
    k: float = Thresholds.competitor_k,
) -> list[Recommendation]:
    """Проставляет приоритет и сортирует по убыванию.

    Рекомендации без проблемы (про сильные стороны для карточки товара)
    получают приоритет по позитиву и идут после рекомендаций по проблемам
    с тем же значением.
    """
    issue_by_id = {i.id: i for i in issues}
    flags = competitor_better_by_issue or {}
    ranked = []
    for rec in recommendations:
        issue = issue_by_id.get(rec.issue_id or "")
        if issue is None:
            p = 0.0
        else:
            p = priority(issue.coverage, issue.neg_share, rec.effort, flags.get(issue.id, False), k)
        ranked.append(rec.model_copy(update={"priority": round(p, 6)}))
    return sorted(ranked, key=lambda r: (-(r.priority or 0), r.effort, r.id))
