"""Шаг 5: агрегация упоминаний в таблицу «аспект × тональность» (без вызовов модели).

Нужны файл с отзывами и mentions.json из scripts/run_extract.py. Пример:
    python scripts/run_aggregate.py data/synthetic_reviews.csv --product "Куртка" \
        --mentions data/mentions.json --taxonomy data/taxonomy_synthetic_v2.json
Токены не тратятся: модель не вызывается.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from review_agent.ingest.loader import загрузить  # noqa: E402
from review_agent.pipeline.aggregate import агрегировать, сохранить_csv  # noqa: E402
from review_agent.pipeline.extract import Упоминание  # noqa: E402
from review_agent.pipeline.taxonomy import Справочник  # noqa: E402
from review_agent.preprocess.cleaner import предобработать  # noqa: E402


def main(аргументы: list[str] | None = None) -> int:
    р = argparse.ArgumentParser(description="Агрегировать упоминания аспектов")
    р.add_argument("файл", help="CSV или XLSX с отзывами (те же, что при извлечении)")
    р.add_argument("--product", required=True)
    р.add_argument("--mentions", default="data/mentions.json")
    р.add_argument("--taxonomy", help="JSON справочника: задаёт порядок атрибутов в таблице")
    р.add_argument("--out", default="data/aggregate.csv")
    а = р.parse_args(аргументы)

    for путь in (а.файл, а.mentions):
        if not Path(путь).is_file():
            print(f"Файл не найден: {путь}", file=sys.stderr)
            return 2

    отзывы = предобработать(загрузить(а.файл, product=а.product)).reviews
    упоминания = [Упоминание(**д) for д in json.loads(Path(а.mentions).read_text(encoding="utf-8"))]
    порядок = [x.name for x in Справочник.загрузить(а.taxonomy).attributes] if а.taxonomy else None
    статистика = агрегировать(отзывы, упоминания, порядок)
    сохранить_csv(статистика, а.out)

    print(f"Отзывов: {len(отзывы)}, упоминаний: {len(упоминания)}. Таблица сохранена в {а.out}\n")
    print(f"{'Аспект':32} {'упом.':>5} {'охват':>6} {'+':>3} {'-':>3} {'=':>3} {'негатив':>8} {'оценка':>7}")
    for с in статистика:
        оценка = "-" if с.mean_rating is None else f"{с.mean_rating:.1f}"
        print(f"{с.attribute[:32]:32} {с.mentions:>5} {с.coverage:>6.0%} {с.positive:>3} "
              f"{с.negative:>3} {с.neutral:>3} {с.negative_share:>8.0%} {оценка:>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
