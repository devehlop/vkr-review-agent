"""Шаг 7 конвейера, расчётная часть: сравнение своего товара с конкурентами по атрибутам (FR-12)."""
from dataclasses import dataclass, field

from review_agent.pipeline.aggregate import AspectStats

WORSE, BETTER, SAME, NO_DATA = "хуже", "лучше", "сопоставимо", "мало данных"


@dataclass
class Comparison:
    attribute: str
    own_neg_share: float
    own_coverage: float
    competitors: dict[str, tuple[float, float]] = field(default_factory=dict)  # товар -> (доля негатива, охват)
    verdict: str = NO_DATA  # свой товар относительно среднего по конкурентам
    rarer_at_competitors: bool = False  # негатив у конкурентов заметно реже: реальное отставание


def compare(
    stats: list[AspectStats], own_product: str, min_mentions: int = 5, threshold: float = 0.10
) -> list[Comparison]:
    """Сравнивает долю негатива по каждому атрибуту.

    Атрибут сравнивается, только если у своего товара и хотя бы у одного конкурента
    не меньше min_mentions упоминаний: на малых числах доли случайны.
    threshold: разница долей негатива, начиная с которой товары считаются разными.
    """
    by_attribute: dict[str, dict[str, AspectStats]] = {}
    for item in stats:
        if not item.feature:
            by_attribute.setdefault(item.attribute, {})[item.product] = item

    result = []
    for attribute, products in by_attribute.items():
        own = products.get(own_product)
        if own is None:
            continue
        row = Comparison(attribute, own.neg_share, own.coverage)
        reliable = []
        for name, item in products.items():
            if name == own_product:
                continue
            row.competitors[name] = (item.neg_share, item.coverage)
            if item.mentions >= min_mentions:
                reliable.append(item.neg_share)
        if own.mentions >= min_mentions and reliable:
            diff = own.neg_share - sum(reliable) / len(reliable)
            row.verdict = WORSE if diff >= threshold else BETTER if diff <= -threshold else SAME
            row.rarer_at_competitors = row.verdict == WORSE
        result.append(row)
    result.sort(key=lambda r: (r.verdict != WORSE, -r.own_neg_share * r.own_coverage))
    return result
