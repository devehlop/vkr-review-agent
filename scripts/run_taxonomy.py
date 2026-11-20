"""Проверка справочника аспектов на реальной выборке отзывов.

Конвейер: загрузка CSV/XLSX → предобработка → справочник аспектов (GigaChat) → JSON.

Пример:
    export GIGACHAT_CREDENTIALS=...   # ключ авторизации, не вставлять в команду
    export GIGACHAT_CA_BUNDLE=~/Downloads/russian_ca_bundle.pem
    python scripts/run_taxonomy.py data/reviews.csv --product "Куртка" --category "женская одежда"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from review_agent.ingest.loader import загрузить  # noqa: E402
from review_agent.llm.gateway import LLMGateway  # noqa: E402
from review_agent.llm.gigachat import GigaChatClient  # noqa: E402
from review_agent.pipeline.taxonomy import построить_справочник  # noqa: E402
from review_agent.preprocess.cleaner import предобработать  # noqa: E402


def main(аргументы: list[str] | None = None, клиент=None) -> int:
    р = argparse.ArgumentParser(description="Построить справочник аспектов по отзывам")
    р.add_argument("файл", help="CSV или XLSX с отзывами (без персональных данных)")
    р.add_argument("--product", required=True, help="название товара")
    р.add_argument("--category", required=True, help="категория, например «женская одежда»")
    р.add_argument("--out", default="data/taxonomy.json", help="куда сохранить справочник")
    р.add_argument("--sample", type=int, default=150, help="размер выборки отзывов для модели")
    р.add_argument("--seed", type=int, default=42)
    р.add_argument("--model", default="GigaChat-2-Pro")
    р.add_argument("--db", default=".cache/llm.sqlite", help="кэш и журнал вызовов модели")
    а = р.parse_args(аргументы)

    if not Path(а.файл).is_file():
        print(f"Файл не найден: {а.файл}. Укажите путь к вашей выгрузке отзывов (CSV или XLSX).",
              file=sys.stderr)
        return 2

    отзывы = загрузить(а.файл, product=а.product)
    результат = предобработать(отзывы)
    print("Предобработка:", результат.stats)
    if not результат.reviews:
        print("После очистки не осталось отзывов.", file=sys.stderr)
        return 1

    Path(а.db).parent.mkdir(parents=True, exist_ok=True)
    шлюз = LLMGateway(клиент or GigaChatClient(model=а.model), db_path=а.db)
    справочник = построить_справочник(шлюз, результат.reviews, категория=а.category,
                                      размер_выборки=а.sample, seed=а.seed)

    Path(а.out).parent.mkdir(parents=True, exist_ok=True)
    справочник.сохранить(а.out)
    print(f"\nСправочник ({len(справочник.attributes)} атрибутов), сохранён в {а.out}:")
    for атрибут in справочник.attributes:
        print(f"  • {атрибут.name}: {', '.join(атрибут.features)}")
    s = шлюз.stats()
    print(f"\nВызовов: {s['calls']} (из кэша {s['cache_hits']}, ошибок {s['errors']}), "
          f"токенов: {s['tokens_in']} + {s['tokens_out']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
