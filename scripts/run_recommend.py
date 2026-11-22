"""Шаги 8–9: рекомендации по проблемам и цикл «генератор — критик».

Нужны: файл с отзывами (для числа отзывов), issues.json (run_issues.py), справочник
(категория) и, по желанию, aggregate.csv (run_aggregate.py) для доли негатива.
Пример:
    python scripts/run_recommend.py data/synthetic_reviews.csv --product "Куртка" \
        --taxonomy data/taxonomy_synthetic_v2.json --issues data/issues.json \
        --aggregate data/aggregate.csv
Флаг --no-critic запускает генератор без проверки: так оценивается вклад критика.
С --mentions data/mentions.json добавляются рекомендации по сильным сторонам для карточки
товара (темы, по которым есть проблемы, в сильные стороны не попадают).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from review_agent.ingest.loader import загрузить  # noqa: E402
from review_agent.llm.gateway import LLMGateway  # noqa: E402
from review_agent.llm.gigachat import GigaChatClient  # noqa: E402
from review_agent.pipeline.aggregate import агрегировать  # noqa: E402
from review_agent.pipeline.critic import рекомендации_по_проблемам  # noqa: E402
from review_agent.pipeline.extract import Упоминание  # noqa: E402
from review_agent.pipeline.strengths import найти_сильные_стороны, рекомендации_по_сильным_сторонам  # noqa: E402
from review_agent.pipeline.issues import Проблема  # noqa: E402
from review_agent.pipeline.taxonomy import Справочник  # noqa: E402
from review_agent.preprocess.cleaner import предобработать  # noqa: E402


def доли_негатива(путь: Path, товар: str) -> dict[str, float]:
    """Доля негатива по атрибутам своего товара из aggregate.csv."""
    доли: dict[str, float] = {}
    with путь.open(encoding="utf-8", newline="") as f:
        for строка in csv.DictReader(f):
            if строка["product"] == товар and строка["is_own"] in ("1", "True", "true"):
                доли[строка["attribute"]] = float(строка["negative_share"])
    return доли


def main(аргументы: list[str] | None = None, клиент=None) -> int:
    р = argparse.ArgumentParser(description="Рекомендации по проблемам с проверкой критиком")
    р.add_argument("файл", help="CSV или XLSX с отзывами")
    р.add_argument("--product", required=True)
    р.add_argument("--taxonomy", required=True)
    р.add_argument("--issues", default="data/issues.json")
    р.add_argument("--aggregate", default=None, help="aggregate.csv для доли негатива")
    р.add_argument("--out", default="data/recommendations.json")
    р.add_argument("--count", type=int, default=2, help="рекомендаций на проблему")
    р.add_argument("--no-critic", action="store_true", help="без проверки критиком")
    р.add_argument("--mentions", default=None,
                   help="mentions.json: с ним добавляются рекомендации по сильным сторонам для карточки")
    р.add_argument("--strengths", type=int, default=3, help="сколько сильных сторон взять (с --mentions)")
    р.add_argument("--model", default="GigaChat-2-Pro")
    р.add_argument("--db", default=".cache/llm.sqlite")
    а = р.parse_args(аргументы)

    for путь in filter(None, (а.файл, а.taxonomy, а.issues, а.aggregate, а.mentions)):
        if not Path(путь).is_file():
            print(f"Файл не найден: {путь}", file=sys.stderr)
            return 2

    справочник = Справочник.загрузить(а.taxonomy)
    отзывы = предобработать(загрузить(а.файл, product=а.product)).reviews
    проблемы = [Проблема(**д) for д in json.loads(Path(а.issues).read_text(encoding="utf-8"))]
    доли = доли_негатива(Path(а.aggregate), а.product) if а.aggregate else {}

    Path(а.db).parent.mkdir(parents=True, exist_ok=True)
    шлюз = LLMGateway(клиент or GigaChatClient(model=а.model), db_path=а.db)
    итог = рекомендации_по_проблемам(шлюз, проблемы, категория=справочник.category,
                                     всего_отзывов=len(отзывы), доли_негатива=доли,
                                     с_критиком=not а.no_critic, количество=а.count)

    сильные = None
    if а.mentions and а.strengths:
        упоминания = [Упоминание(**д) for д in json.loads(Path(а.mentions).read_text(encoding="utf-8"))]
        стороны = найти_сильные_стороны(агрегировать(отзывы, упоминания), упоминания, лимит=а.strengths,
                                        исключить={п.attribute for п in проблемы})
        сильные = рекомендации_по_сильным_сторонам(
            шлюз, стороны, категория=справочник.category, всего_отзывов=len(отзывы),
            принятые=[п.recommendation.action for п in итог.accepted], с_критиком=not а.no_critic)

    результат = {"stats": итог.stats, "accepted": [asdict(п) for п in итог.accepted],
                 "rejected": [asdict(п) for п in итог.rejected],
                 "problems_without_recommendation": итог.problems_without_recommendation}
    if сильные is not None:
        результат["strengths"] = {"stats": сильные.stats, "accepted": [asdict(п) for п in сильные.accepted],
                                  "rejected": [asdict(п) for п in сильные.rejected],
                                  "without_recommendation": сильные.problems_without_recommendation}
    Path(а.out).parent.mkdir(parents=True, exist_ok=True)
    Path(а.out).write_text(json.dumps(результат, ensure_ascii=False, indent=1), encoding="utf-8")

    print("Рекомендации:", итог.stats, "\n")
    for i, п in enumerate(итог.accepted, 1):
        р_ = п.recommendation
        оценки = " ".join(str(в) for в in п.verdicts[-1].scores.values()) if п.verdicts else "без критика"
        print(f"{i}. [{р_.problem}] {р_.action}\n     тип: {р_.type}, сложность {р_.effort}, "
              f"оценки критика: {оценки}, доработок: {max(0, len(п.verdicts) - 1)}")
    if итог.rejected:
        print("\nОтклонены:")
        for п in итог.rejected:
            print(f"  - [{п.recommendation.problem}] {п.recommendation.action} ({п.reason})")
    if итог.problems_without_recommendation:
        print("\nПроблемы без рекомендации:", "; ".join(итог.problems_without_recommendation))
    if сильные is not None:
        print("\nСильные стороны:", сильные.stats)
        for i, п in enumerate(сильные.accepted, 1):
            print(f"{i}. [{п.recommendation.problem}] {п.recommendation.action}")
        for п in сильные.rejected:
            print(f"  - отклонена [{п.recommendation.problem}] {п.recommendation.action} ({п.reason})")
    for шаг, ошибка in шлюз.ошибки():
        print(f"  ошибка [{шаг}]: {ошибка}")
    s = шлюз.stats()
    print(f"\nСохранено в {а.out}. Вызовов: {s['calls']} (из кэша {s['cache_hits']}, "
          f"ошибок {s['errors']}), токенов: {s['tokens_in']} + {s['tokens_out']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
