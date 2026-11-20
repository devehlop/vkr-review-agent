"""Проверка шага 4: извлечение аспектов и тональности по предложениям.

Нужен справочник из scripts/run_taxonomy.py. Пример:
    python scripts/run_extract.py data/synthetic_reviews.csv --product "Куртка" \
        --taxonomy data/taxonomy_synthetic_v2.json
Ключ и сертификат: переменные GIGACHAT_CREDENTIALS и GIGACHAT_CA_BUNDLE.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from review_agent.ingest.loader import загрузить  # noqa: E402
from review_agent.llm.gateway import LLMGateway  # noqa: E402
from review_agent.llm.gigachat import GigaChatClient  # noqa: E402
from review_agent.pipeline.extract import извлечь_аспекты  # noqa: E402
from review_agent.pipeline.taxonomy import Справочник  # noqa: E402
from review_agent.preprocess.cleaner import предобработать  # noqa: E402


def main(аргументы: list[str] | None = None, клиент=None) -> int:
    р = argparse.ArgumentParser(description="Извлечь аспекты и тональность из отзывов")
    р.add_argument("файл", help="CSV или XLSX с отзывами")
    р.add_argument("--product", required=True)
    р.add_argument("--taxonomy", required=True, help="JSON справочника (run_taxonomy.py)")
    р.add_argument("--out", default="data/mentions.json")
    р.add_argument("--batch", type=int, default=25, help="предложений в одном запросе")
    р.add_argument("--limit", type=int, default=0, help="обработать только первые N предложений")
    р.add_argument("--model", default="GigaChat-2-Pro")
    р.add_argument("--db", default=".cache/llm.sqlite")
    а = р.parse_args(аргументы)

    for путь in (а.файл, а.taxonomy):
        if not Path(путь).is_file():
            print(f"Файл не найден: {путь}", file=sys.stderr)
            return 2

    справочник = Справочник.загрузить(а.taxonomy)
    результат = предобработать(загрузить(а.файл, product=а.product))
    предложения = результат.sentences[:а.limit] if а.limit else результат.sentences
    print("Предобработка:", результат.stats)
    print(f"В обработку: {len(предложения)} предложений, пакетами по {а.batch}")

    Path(а.db).parent.mkdir(parents=True, exist_ok=True)
    шлюз = LLMGateway(клиент or GigaChatClient(model=а.model), db_path=а.db)
    итог = извлечь_аспекты(шлюз, предложения, справочник, размер_пакета=а.batch)

    Path(а.out).parent.mkdir(parents=True, exist_ok=True)
    Path(а.out).write_text(json.dumps([м.__dict__ for м in итог.mentions], ensure_ascii=False, indent=1),
                           encoding="utf-8")
    print("\nИзвлечение:", итог.stats)
    счёт = Counter((м.attribute, м.sentiment) for м in итог.mentions)
    print("\nУпоминаний по аспектам (положительные / отрицательные / нейтральные):")
    for атрибут in справочник.attributes:
        п, о, н = (счёт[(атрибут.name, т)] for т in ("positive", "negative", "neutral"))
        print(f"  {атрибут.name}: +{п} / -{о} / ={н}")
    for шаг, ошибка in шлюз.ошибки():
        print(f"  ошибка [{шаг}]: {ошибка}")
    s = шлюз.stats()
    print(f"\nСохранено в {а.out}. Вызовов: {s['calls']} (из кэша {s['cache_hits']}, "
          f"ошибок {s['errors']}), токенов: {s['tokens_in']} + {s['tokens_out']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
