"""Шаг 10 конвейера: приоритизация рекомендаций по объяснимой формуле (FR-16)."""
from review_agent.models import Issue, Recommendation

COMPETITOR_FACTOR = 1.5  # коэффициент, если у конкурентов проблема встречается заметно реже


def priority(coverage: float, neg_share: float, effort: int, rarer_at_competitors: bool = False,
             competitor_factor: float = COMPETITOR_FACTOR) -> float:
    """P = охват × доля негатива × K_конк / сложность.

    Чем чаще встречается проблема и чем проще её исправить, тем выше приоритет.
    """
    if effort not in (1, 2, 3):
        raise ValueError(f"Сложность должна быть 1, 2 или 3, получено: {effort}")
    factor = competitor_factor if rarer_at_competitors else 1.0
    return round(coverage * neg_share * factor / effort, 4)


def prioritize(recommendations: list[Recommendation], issues: list[Issue]) -> list[Recommendation]:
    """Проставляет приоритет и сортирует рекомендации по убыванию."""
    issue_by_id = {i.id: i for i in issues}
    for item in recommendations:
        issue = issue_by_id.get(item.issue_id)
        item.priority = priority(issue.coverage, issue.neg_share, item.effort, issue.rarer_at_competitors) if issue else 0.0
    return sorted(recommendations, key=lambda r: (-r.priority, r.id))
