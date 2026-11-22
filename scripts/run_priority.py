"""Шаги 11 и 10: проверка опоры рекомендаций на отзывы и приоритизация. Без LLM.

Нужны: файл с отзывами, issues.json (run_issues.py), recommendations.json
(run_recommend.py) и, по желанию, aggregate.csv (run_aggregate.py) для доли негатива.
Пример:
    python scripts/run_priority.py data/synthetic_reviews.csv --product "Куртка" \
        --issues data/issues.json --recommendations data/recommendations.json \
        --aggregate data/aggregate.csv
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
from review_agent.pipeline.grounding import проверить_опору  # noqa: E402
from review_agent.pipeline.issues import Проблема  # noqa: E402
from review_agent.pipeline.prioritize import расставить  # noqa: E402
from review_agent.pipeline.recommend import Рекомендация  # noqa: E402
from review_agent.preprocess.cleaner import предобработать  # noqa: E402


def доли_негатива(путь: Path, товар: str) -> dict[str, float]:
    """Доля негатива по атрибутам своего товара из aggregate.csv."""
    доли: dict[str, float] = {}
    with путь.open(encoding="utf-8", newline="") as f:
        for строка in csv.DictReader(f):
            if строка["product"] == товар and строка["is_own"] in ("1", "True", "true"):
                доли[строка["attribute"]] = float(строка["negative_share"])
    return доли


def main(аргументы: list[str] | None = None) -> int:
    р = argparse.ArgumentParser(description="Проверка опоры на отзывы и приоритеты рекомендаций")
    р.add_argument("файл", help="CSV или XLSX с отзывами")
    р.add_argument("--product", required=True)
    р.add_argument("--issues", default="data/issues.json")
    р.add_argument("--recommendations", default="data/recommendations.json")
    р.add_argument("--aggregate", default=None, help="aggregate.csv для доли негатива")
    р.add_argument("--out", default="data/prioritized.json")
    а = р.parse_args(аргументы)

    for путь in filter(None, (а.файл, а.issues, а.recommendations, а.aggregate)):
        if not Path(путь).is_file():
            print(f"Файл не найден: {путь}", file=sys.stderr)
            return 2

    отзывы = предобработать(загрузить(а.файл, product=а.product)).reviews
    проблемы = [Проблема(**д) for д in json.loads(Path(а.issues).read_text(encoding="utf-8"))]
    данные = json.loads(Path(а.recommendations).read_text(encoding="utf-8"))
    принятые = [Рекомендация(**п["recommendation"]) for п in данные.get("accepted", [])]
    доли = доли_негатива(Path(а.aggregate), а.product) if а.aggregate else {}

    прошли, отказы = проверить_опору(принятые, проблемы, отзывы)
    итог = расставить(прошли, проблемы, доли_негатива=доли)
    без_доли = sorted({п.recommendation.attribute for п in итог if not п.negative_share_known})

    stats = {"recommendations": len(принятые), "grounded": len(прошли), "ungrounded": len(отказы),
             "critic": данные.get("stats", {}).get("critic"), "negative_share_missing": без_доли}
    Path(а.out).parent.mkdir(parents=True, exist_ok=True)
    Path(а.out).write_text(json.dumps(
        {"stats": stats, "prioritized": [asdict(п) for п in итог],
         "ungrounded": [asdict(о) for о in отказы]}, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"Рекомендаций: {len(принятые)}, прошли проверку опоры: {len(прошли)}, "
          f"отклонены: {len(отказы)}\n")
    for i, п in enumerate(итог, 1):
        р_ = п.recommendation
        доля = f"{п.negative_share:.0%}" if п.negative_share_known else "нет данных"
        print(f"{i}. P = {п.priority:.4f}  {р_.action}\n     проблема: {р_.problem}; охват {п.coverage:.0%}, "
              f"доля негатива {доля}, сложность {р_.effort}")
    for о in отказы:
        print(f"  отклонена: [{о.problem}] {о.action} ({о.reason})")
    if без_доли:
        print("\nНет доли негатива для атрибутов:", ", ".join(без_доли),
              "— передайте --aggregate, иначе в формуле берётся 1,0.")
    print(f"\nСохранено в {а.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
