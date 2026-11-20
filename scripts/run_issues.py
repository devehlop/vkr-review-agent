"""Шаг 6: выделение проблем своего товара по негативным упоминаниям.

Нужны: файл с отзывами, mentions.json (run_extract.py) и справочник (категория).
Вызывает GigaChat: эмбеддинги цитат и по одному запросу на название каждой проблемы.
Пример:
    python scripts/run_issues.py data/synthetic_reviews.csv --product "Куртка" \
        --taxonomy data/taxonomy_synthetic_v2.json --mentions data/mentions.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from review_agent.ingest.loader import загрузить  # noqa: E402
from review_agent.llm.embeddings import GigaChatEmbedder  # noqa: E402
from review_agent.llm.gateway import LLMGateway  # noqa: E402
from review_agent.llm.gigachat import GigaChatClient  # noqa: E402
from review_agent.pipeline.extract import Упоминание  # noqa: E402
from review_agent.pipeline.issues import выделить_проблемы  # noqa: E402
from review_agent.pipeline.taxonomy import Справочник  # noqa: E402
from review_agent.preprocess.cleaner import предобработать  # noqa: E402


def main(аргументы: list[str] | None = None, клиент=None, эмбеддер=None) -> int:
    р = argparse.ArgumentParser(description="Выделить проблемы по негативным упоминаниям")
    р.add_argument("файл", help="CSV или XLSX с отзывами")
    р.add_argument("--product", required=True)
    р.add_argument("--taxonomy", required=True)
    р.add_argument("--mentions", default="data/mentions.json")
    р.add_argument("--out", default="data/issues.json")
    р.add_argument("--threshold", type=float, default=0.35, help="порог косинусного расстояния")
    р.add_argument("--min-size", type=int, default=3, help="минимум упоминаний в проблеме")
    р.add_argument("--model", default="GigaChat-2-Pro")
    р.add_argument("--embeddings-model", default="Embeddings")
    р.add_argument("--db", default=".cache/llm.sqlite")
    а = р.parse_args(аргументы)

    for путь in (а.файл, а.taxonomy, а.mentions):
        if not Path(путь).is_file():
            print(f"Файл не найден: {путь}", file=sys.stderr)
            return 2

    справочник = Справочник.загрузить(а.taxonomy)
    отзывы = предобработать(загрузить(а.файл, product=а.product)).reviews
    упоминания = [Упоминание(**д) for д in json.loads(Path(а.mentions).read_text(encoding="utf-8"))]

    Path(а.db).parent.mkdir(parents=True, exist_ok=True)
    gc = клиент or GigaChatClient(model=а.model)
    шлюз = LLMGateway(gc, db_path=а.db)
    итог = выделить_проблемы(шлюз, эмбеддер or GigaChatEmbedder(gc, а.embeddings_model), упоминания,
                             категория=справочник.category, всего_отзывов=len(отзывы),
                             порог=а.threshold, мин_размер=а.min_size)

    Path(а.out).parent.mkdir(parents=True, exist_ok=True)
    Path(а.out).write_text(json.dumps([asdict(п) for п in итог.problems], ensure_ascii=False, indent=1),
                           encoding="utf-8")
    print("Проблемы:", итог.stats, "\n")
    for i, п in enumerate(итог.problems, 1):
        print(f"{i}. [{п.attribute}] {п.title}  (отзывов {п.reviews}, охват {п.coverage:.0%}, "
              f"упоминаний {п.mentions})")
        for ц in п.quotes[:3]:
            print(f"     «{ц}»")
    for шаг, ошибка in шлюз.ошибки():
        print(f"  ошибка [{шаг}]: {ошибка}")
    s = шлюз.stats()
    print(f"\nСохранено в {а.out}. Вызовов: {s['calls']} (из кэша {s['cache_hits']}, "
          f"ошибок {s['errors']}), токенов: {s['tokens_in']} + {s['tokens_out']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
