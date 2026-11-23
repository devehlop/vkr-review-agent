"""Шаг 12: отчёт для продавца в Markdown и DOCX. Без LLM.

Нужны файлы предыдущих шагов: отзывы, aggregate.csv (шаг 5), issues.json (шаг 6),
prioritized.json (шаги 10–11) и, по желанию, recommendations.json (шаги 8–9) —
с ним в отчёт попадают советы по сильным сторонам и отклонённые критиком рекомендации.
Пример:
    python scripts/run_report.py data/synthetic_reviews.csv --product "Куртка" \
        --aggregate data/aggregate.csv --issues data/issues.json \
        --prioritized data/prioritized.json --recommendations data/recommendations.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from review_agent.ingest.loader import загрузить  # noqa: E402
from review_agent.preprocess.cleaner import предобработать  # noqa: E402
from review_agent.report.build import собрать_из_файлов  # noqa: E402
from review_agent.report.render import в_docx, в_markdown  # noqa: E402


def main(аргументы: list[str] | None = None) -> int:
    р = argparse.ArgumentParser(description="Отчёт для продавца в Markdown и DOCX")
    р.add_argument("файл", help="CSV или XLSX с отзывами")
    р.add_argument("--product", required=True)
    р.add_argument("--aggregate", default="data/aggregate.csv")
    р.add_argument("--issues", default="data/issues.json")
    р.add_argument("--prioritized", default="data/prioritized.json")
    р.add_argument("--recommendations", default=None)
    р.add_argument("--out", default="data/report", help="путь без расширения: будут .md и .docx")
    а = р.parse_args(аргументы)

    for путь in filter(None, (а.файл, а.aggregate, а.issues, а.prioritized, а.recommendations)):
        if not Path(путь).is_file():
            print(f"Файл не найден: {путь}", file=sys.stderr)
            return 2

    предобработка = предобработать(загрузить(а.файл, product=а.product))
    отчёт = собрать_из_файлов(
        товар=а.product, отзывов=len(предобработка.reviews),
        подозрительных=len(предобработка.suspicious_ids), агрегат=а.aggregate, проблемы=а.issues,
        приоритеты=а.prioritized, рекомендации=а.recommendations)

    md = Path(а.out).with_suffix(".md")
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(в_markdown(отчёт), encoding="utf-8")
    docx = в_docx(отчёт, Path(а.out).with_suffix(".docx"))
    print(f"Отчёт: {md} и {docx}. Рекомендаций: {len(отчёт.recommendations)}, "
          f"проблем: {len(отчёт.problems)}, сильных сторон: {len(отчёт.strengths)}, "
          f"отклонено: {len(отчёт.rejected)}.")
    for п in отчёт.warnings:
        print("Предупреждение:", п)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
