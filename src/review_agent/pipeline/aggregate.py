"""Шаг 5: агрегация без LLM (FR-8, FR-9, docs/architecture.md).

Для каждого товара и каждого атрибута считаются:
- число упоминаний и охват (доля отзывов товара, где атрибут упомянут);
- распределение тональности и доля негатива среди упоминаний;
- средняя оценка отзывов, где атрибут упомянут;
- динамика доли негатива по месяцам (по дате отзыва).

Шаг извлечения даёт только уровень «атрибут»; признаки внутри атрибута пока не
выделяются, поэтому агрегация по признакам появится вместе с ними.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from review_agent.ingest.loader import Отзыв
from review_agent.pipeline.extract import Упоминание


@dataclass
class СтатАспекта:
    product: str
    is_own: bool
    attribute: str
    mentions: int = 0
    reviews: int = 0                # отзывов товара, где атрибут упомянут
    coverage: float = 0.0           # reviews / все отзывы товара
    positive: int = 0
    negative: int = 0
    neutral: int = 0
    negative_share: float = 0.0     # negative / mentions
    mean_rating: float | None = None
    # месяц "ГГГГ-ММ" -> (упоминаний, доля негатива); только отзывы с датой
    by_month: dict[str, tuple[int, float]] = field(default_factory=dict)


def агрегировать(отзывы: list[Отзыв], упоминания: list[Упоминание],
                 порядок_атрибутов: list[str] | None = None) -> list[СтатАспекта]:
    """Считает статистику по каждому (товар, атрибут), у которого есть упоминания.

    порядок_атрибутов: порядок атрибутов в результате (например, из справочника);
    остальные идут в алфавитном порядке. Сначала свой товар, затем конкуренты.
    """
    по_id = {(о.product, о.id): о for о in отзывы}
    всего: dict[tuple[str, bool], int] = defaultdict(int)
    for о in отзывы:
        всего[(о.product, о.is_own)] += 1

    группы: dict[tuple[str, bool, str], list[Упоминание]] = defaultdict(list)
    for м in упоминания:
        группы[(м.product, м.is_own, м.attribute)].append(м)

    порядок = {имя: i for i, имя in enumerate(порядок_атрибутов or [])}
    результат: list[СтатАспекта] = []
    for (товар, свой, атрибут), список in group_sorted(группы, порядок):
        id_отзывов = {м.review_id for м in список}
        связанные = [по_id[(товар, i)] for i in id_отзывов if (товар, i) in по_id]
        с_оценкой = [о.rating for о in связанные if о.rating is not None]
        ст = СтатАспекта(товар, свой, атрибут, mentions=len(список), reviews=len(id_отзывов))
        ст.coverage = ст.reviews / всего[(товар, свой)] if всего[(товар, свой)] else 0.0
        ст.positive = sum(м.sentiment == "positive" for м in список)
        ст.negative = sum(м.sentiment == "negative" for м in список)
        ст.neutral = sum(м.sentiment == "neutral" for м in список)
        ст.negative_share = ст.negative / ст.mentions
        ст.mean_rating = sum(с_оценкой) / len(с_оценкой) if с_оценкой else None

        по_месяцам: dict[str, list[Упоминание]] = defaultdict(list)
        for м in список:
            отзыв = по_id.get((товар, м.review_id))
            if отзыв and отзыв.date:
                по_месяцам[отзыв.date.strftime("%Y-%m")].append(м)
        ст.by_month = {
            месяц: (len(ms), sum(x.sentiment == "negative" for x in ms) / len(ms))
            for месяц, ms in sorted(по_месяцам.items())}
        результат.append(ст)
    return результат


def group_sorted(группы: dict, порядок: dict[str, int]):
    """Свой товар первым, затем по названию товара, затем по порядку атрибутов."""
    return sorted(группы.items(),
                  key=lambda kv: (not kv[0][1], kv[0][0], порядок.get(kv[0][2], 10_000), kv[0][2]))


def сохранить_csv(статистика: list[СтатАспекта], путь: str | Path) -> None:
    """Плоская таблица для отчёта и проверки (динамика по месяцам в одну колонку)."""
    Path(путь).parent.mkdir(parents=True, exist_ok=True)
    with open(путь, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["product", "is_own", "attribute", "mentions", "reviews", "coverage",
                    "positive", "negative", "neutral", "negative_share", "mean_rating", "by_month"])
        for с in статистика:
            динамика = "; ".join(f"{м}: {n} / {доля:.0%}" for м, (n, доля) in с.by_month.items())
            w.writerow([с.product, int(с.is_own), с.attribute, с.mentions, с.reviews,
                        f"{с.coverage:.3f}", с.positive, с.negative, с.neutral,
                        f"{с.negative_share:.3f}",
                        "" if с.mean_rating is None else f"{с.mean_rating:.2f}", динамика])
