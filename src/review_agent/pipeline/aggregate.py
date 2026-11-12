"""Шаг 5 конвейера: агрегация упоминаний без LLM (FR-8, FR-9)."""
from dataclasses import dataclass, field

from review_agent.models import Review, Sentence
from review_agent.pipeline.extract import Mention


@dataclass
class AspectStats:
    """Статистика по атрибуту (feature пустой) или по признаку внутри атрибута."""

    product: str
    attribute: str
    feature: str = ""
    mentions: int = 0
    reviews: int = 0  # число отзывов с упоминанием
    coverage: float = 0.0  # охват: доля отзывов товара с упоминанием
    negative: int = 0
    neutral: int = 0
    positive: int = 0
    neg_share: float = 0.0  # доля негативных упоминаний
    avg_rating: float | None = None  # средняя оценка отзывов с упоминанием
    by_month: dict[str, tuple[int, int]] = field(default_factory=dict)  # «2026-09» -> (негативных, всего)


def aggregate(
    reviews: list[Review], sentences: list[Sentence], mentions: dict[int, list[Mention]]
) -> list[AspectStats]:
    """Считает статистику по каждому товару, атрибуту и признаку.

    mentions: номер предложения -> упоминания (результат шага 4).
    """
    review_by_id = {r.id: r for r in reviews}
    total_reviews: dict[str, int] = {}
    for review in reviews:
        total_reviews[review.product] = total_reviews.get(review.product, 0) + 1

    # ключ (товар, атрибут, признак) -> список (отзыв, тональность)
    groups: dict[tuple[str, str, str], list[tuple[Review, int]]] = {}
    for sentence in sentences:
        review = review_by_id.get(sentence.review_id)
        if review is None:
            continue
        for mention in mentions.get(sentence.id, []):
            feature = mention.feature.strip().lower()
            groups.setdefault((review.product, mention.attribute, ""), []).append((review, mention.sentiment))
            if feature:
                groups.setdefault((review.product, mention.attribute, feature), []).append((review, mention.sentiment))

    result = []
    for (product, attribute, feature), items in groups.items():
        stats = AspectStats(product, attribute, feature, mentions=len(items))
        stats.negative = sum(s == -1 for _, s in items)
        stats.neutral = sum(s == 0 for _, s in items)
        stats.positive = sum(s == 1 for _, s in items)
        stats.neg_share = round(stats.negative / stats.mentions, 4)
        unique = {r.id: r for r, _ in items}
        stats.reviews = len(unique)
        stats.coverage = round(stats.reviews / total_reviews[product], 4)
        ratings = [r.rating for r in unique.values() if r.rating is not None]
        stats.avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else None
        for review, sentiment in items:
            if review.created:
                month = review.created.strftime("%Y-%m")
                negative, total = stats.by_month.get(month, (0, 0))
                stats.by_month[month] = (negative + (sentiment == -1), total + 1)
        stats.by_month = dict(sorted(stats.by_month.items()))
        result.append(stats)
    # сначала атрибуты с наибольшим числом упоминаний
    result.sort(key=lambda s: (s.product, s.feature != "", -s.mentions, s.attribute, s.feature))
    return result
