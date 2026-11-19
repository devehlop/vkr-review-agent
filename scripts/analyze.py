"""Анализ отзывов из файлов и отчёт в Markdown.

Примеры:
    python scripts/analyze.py --own data/samples/demo_own.csv --competitor data/samples/demo_competitor.csv
    python scripts/analyze.py --own отзывы.xlsx --provider gigachat --taxonomy auto

По умолчанию работает демо-заглушка (без ключей) и встроенный справочник
clothing_v1. Отчёт и база запусков — в runs/.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from review_agent.config import Settings  # noqa: E402
from review_agent.llm import LLMGateway, make_client  # noqa: E402
from review_agent.llm.demo import demo_client  # noqa: E402
from review_agent.models import Product  # noqa: E402
from review_agent.pipeline.embeddings import SentenceTransformerEmbedder, TfidfEmbedder  # noqa: E402
from review_agent.pipeline.orchestrator import ProductInput, run_analysis  # noqa: E402
from review_agent.report.markdown import render_markdown  # noqa: E402
from review_agent.storage.db import Storage  # noqa: E402
from review_agent.taxonomy import load_taxonomy  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Анализ отзывов и рекомендации продавцу")
    ap.add_argument("--own", required=True, help="файл отзывов своего товара (CSV/XLSX)")
    ap.add_argument("--own-name", default="Свой товар")
    ap.add_argument("--competitor", action="append", default=[], help="файл отзывов конкурента, можно несколько")
    ap.add_argument("--provider", default="demo", help="demo, gigachat, yandex, ollama")
    ap.add_argument("--model", default="")
    ap.add_argument("--taxonomy", default="clothing_v1", help="имя встроенного справочника или auto")
    ap.add_argument("--embedder", default="tfidf", help="tfidf или имя модели sentence-transformers, например cointegrated/rubert-tiny2")
    ap.add_argument("--no-critic", action="store_true", help="без агента-критика (для сравнения)")
    args = ap.parse_args()

    settings = Settings.from_env()
    client = demo_client() if args.provider == "demo" else make_client(Settings(llm_provider=args.provider, llm_model=args.model))
    gateway = LLMGateway(client, temperature=settings.temperature)
    inputs = [ProductInput(Product(id="own", name=args.own_name), path=Path(args.own))]
    for k, path in enumerate(args.competitor, start=1):
        inputs.append(ProductInput(Product(id=f"comp{k}", name=f"Конкурент {k}", is_own=False), path=Path(path)))
    taxonomy = None if args.taxonomy == "auto" else load_taxonomy(args.taxonomy)

    if args.embedder == "tfidf":
        embedder = TfidfEmbedder()
    else:
        embedder = SentenceTransformerEmbedder(args.embedder)

    storage = Storage(ROOT / settings.db_path)
    started = time.perf_counter()
    result = run_analysis(inputs, gateway, settings, taxonomy=taxonomy, storage=storage, embedder=embedder,
                          use_critic=not args.no_critic, progress=lambda s: print(f"… {s}"))
    elapsed = time.perf_counter() - started

    md = render_markdown(result)
    if args.provider == "demo":
        md = "> **Демонстрационный запуск без языковой модели:** аспекты найдены словарным методом, рекомендации шаблонные.\n\n" + md
    out = ROOT / "runs" / "reports" / f"{datetime.now():%Y%m%d-%H%M%S}_{args.provider}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"Готово за {elapsed:.1f} с: {len(result.issues)} проблем, {len(result.recommendations)} рекомендаций")
    print(f"Отчёт: {out.relative_to(ROOT)}; запуск № {result.run_id} в {settings.db_path}")


if __name__ == "__main__":
    main()
