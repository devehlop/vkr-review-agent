"""Запуск анализа из командной строки.

Пример:
    python -m review_agent.cli --own отзывы.xlsx --product "Платье летнее" \\
        --competitor "Платье конкурента=конкурент.csv" --model gigachat --out отчёт.md
"""
import argparse
import json
from pathlib import Path

from review_agent.config import MODEL_NAMES, ROOT, make_client
from review_agent.ingest import load_reviews
from review_agent.llm.gateway import LLMGateway
from review_agent.pipeline.orchestrator import Settings, run_pipeline
from review_agent.report import build_report
from review_agent.storage import Storage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--own", required=True, help="файл CSV/XLSX с отзывами на свой товар")
    parser.add_argument("--product", required=True, help="название своего товара")
    parser.add_argument("--competitor", action="append", default=[], metavar="НАЗВАНИЕ=ФАЙЛ",
                        help="отзывы на товар конкурента, можно указать несколько раз")
    parser.add_argument("--model", default="gigachat", choices=MODEL_NAMES)
    parser.add_argument("--category", default="одежда")
    parser.add_argument("--taxonomy", help="готовый справочник атрибутов (JSON). Без него справочник составит модель")
    parser.add_argument("--no-critic", action="store_true", help="не проверять рекомендации критиком")
    parser.add_argument("--out", default="report.md")
    parser.add_argument("--db", default=str(ROOT / "data" / "review_agent.db"))
    args = parser.parse_args()

    reviews = load_reviews(args.own, product=args.product)
    for item in args.competitor:
        name, _, path = item.partition("=")
        if not path:
            parser.error(f"Конкурент задаётся как НАЗВАНИЕ=ФАЙЛ, получено: {item}")
        reviews += load_reviews(path, product=name, is_own=False)

    client = make_client(args.model)
    gateway = LLMGateway(client, journal_path=ROOT / "logs" / f"{client.name}.jsonl", cache_dir=ROOT / ".cache" / client.name)
    taxonomy = json.loads(Path(args.taxonomy).read_text(encoding="utf-8")) if args.taxonomy else None
    result = run_pipeline(gateway, reviews, args.product, taxonomy,
                          Settings(category=args.category, use_critic=not args.no_critic), Storage(args.db))
    Path(args.out).write_text(build_report(result), encoding="utf-8")

    usage = result.usage
    print(f"Отчёт: {args.out}. Проблем: {len(result.issues)}, рекомендаций: {len(result.recommendations)}, "
          f"отклонено: {len(result.rejected)}.")
    print(f"Запросов к модели: {usage.calls} (из кэша: {usage.cached}), токенов: {usage.tokens_in}/{usage.tokens_out}, "
          f"стоимость: {usage.cost:.2f}, время: {usage.seconds:.0f} с.")
    for warning in result.warnings:
        print("Предупреждение:", warning)


if __name__ == "__main__":
    main()
