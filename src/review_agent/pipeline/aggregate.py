"""Шаг 5. Агрегация упоминаний без LLM (FR-8, FR-9).

Охват — доля отзывов товара, где аспект упомянут. Доля негатива — доля
негативных упоминаний среди всех упоминаний аспекта.
"""

from __future__ import annotations

from collections import defaultdict

from review_agent.models import AspectStats, Mention, Review, Sentence


def _index(reviews: list[Review], sentences: list[Sentence]):
    review_by_id = {r.id: r for r in reviews}
    review_of_sentence = {s.id: s.review_id for s in sentences}
    totals: dict[str, int] = defaultdict(int)
    for r in reviews:
        totals[r.product_id] += 1
    return review_by_id, review_of_sentence, totals


def compute_stats(
    reviews: list[Review],
    sentences: list[Sentence],
    mentions: list[Mention],
    by_feature: bool = False,
) -> list[AspectStats]:
    """Статистика по каждому товару и атрибуту (или паре атрибут + признак)."""
    review_by_id, review_of_sentence, totals = _index(reviews, sentences)
    groups: dict[tuple[str, str, str | None], list[tuple[Mention, Review]]] = defaultdict(list)
    for m in mentions:
        review = review_by_id[review_of_sentence[m.sentence_id]]
        key = (review.product_id, m.attribute, m.feature if by_feature else None)
        groups[key].append((m, review))

    result = []
    for (product_id, attribute, feature), items in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or "")):
        n = len(items)
        reviews_with = {r.id: r for _, r in items}
        ratings = [r.rating for r in reviews_with.values() if r.rating is not None]
        result.append(
            AspectStats(
                product_id=product_id,
                attribute=attribute,
                feature=feature,
                mentions=n,
                reviews_with=len(reviews_with),
                coverage=len(reviews_with) / totals[product_id],
                neg_share=sum(m.sentiment == -1 for m, _ in items) / n,
                pos_share=sum(m.sentiment == 1 for m, _ in items) / n,
                avg_rating=sum(ratings) / len(ratings) if ratings else None,
            )
        )
    return result


def monthly_negative_share(
    reviews: list[Review],
    sentences: list[Sentence],
    mentions: list[Mention],
    product_id: str,
    attribute: str,
) -> dict[str, float]:
    """Динамика доли негатива по месяцам «ГГГГ-ММ» (FR-9). Отзывы без даты не учитываются."""
    review_by_id, review_of_sentence, _ = _index(reviews, sentences)
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for m in mentions:
        r = review_by_id[review_of_sentence[m.sentence_id]]
        if r.product_id != product_id or m.attribute != attribute or r.created is None:
            continue
        month = r.created.strftime("%Y-%m")
        counts[month][0] += m.sentiment == -1
        counts[month][1] += 1
    return {month: neg / total for month, (neg, total) in sorted(counts.items())}
