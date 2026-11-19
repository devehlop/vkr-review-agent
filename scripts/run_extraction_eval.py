"""Мини-эксперимент: качество извлечения аспектов на размеченной выборке.

Примеры запуска из корня репозитория:
    python scripts/run_extraction_eval.py --provider baseline
    python scripts/run_extraction_eval.py --provider fake            # пробный прогон без ключей
    python scripts/run_extraction_eval.py --provider gigachat --model GigaChat-2-Pro
    python scripts/run_extraction_eval.py --provider yandex --model yandexgpt/latest
    python scripts/run_extraction_eval.py --provider ollama --model qwen3:4b

Результаты: runs/eval/<дата>_<выборка>_<провайдер>_<модель>/
    predictions.json — ответы по предложениям;
    metrics.json     — метрики, токены, стоимость, время;
    llm_calls.jsonl  — журнал всех запросов к модели.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from review_agent.config import Settings  # noqa: E402
from review_agent.evaluation import baseline  # noqa: E402
from review_agent.evaluation.annotation import load_annotation, load_sentences  # noqa: E402
from review_agent.evaluation.metrics import evaluate  # noqa: E402
from review_agent.llm import LLMGateway, make_client  # noqa: E402
from review_agent.llm.fake import FakeLLMClient, empty_extraction_responder  # noqa: E402
from review_agent.models import Sentence  # noqa: E402
from review_agent.pipeline.extract import FewShotExample, extract_mentions, mentions_to_labels  # noqa: E402
from review_agent.taxonomy import load_taxonomy  # noqa: E402


def load_fewshot(path: Path) -> list[FewShotExample]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    from review_agent.pipeline.extract import ExtractedMention

    return [FewShotExample(e["sentence"], [ExtractedMention(**m) for m in e["mentions"]]) for e in data["examples"]]


def main() -> None:
    ap = argparse.ArgumentParser(description="Оценка извлечения аспектов на размеченной выборке")
    ap.add_argument("--provider", default="baseline", help="baseline, fake, gigachat, yandex, ollama")
    ap.add_argument("--model", default="")
    ap.add_argument("--split", default="mini", help="имя файла в data/annotation без расширения")
    ap.add_argument("--fewshot", default=str(ROOT / "eval" / "fewshot_examples.json"), help="файл примеров; пусто — без примеров")
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--limit", type=int, default=0, help="взять только первые N предложений")
    args = ap.parse_args()

    taxonomy = load_taxonomy()
    xlsx = ROOT / "data" / "annotation" / f"{args.split}.xlsx"
    gold = load_annotation(xlsx, taxonomy)
    texts = load_sentences(xlsx)
    if args.limit:
        keep = list(texts)[: args.limit]
        texts = {k: texts[k] for k in keep}
        gold = {k: gold[k] for k in keep}
    if not any(gold.values()):
        print("Внимание: в выборке нет ни одной отметки, метрики будут нулевыми (выборка ещё не размечена).")

    started = time.perf_counter()
    summary: dict = {}
    errors: list[str] = []
    if args.provider == "baseline":
        pred = baseline.predict(texts)
        model = "словарный метод"
    else:
        settings = Settings(llm_provider=args.provider, llm_model=args.model)
        client = FakeLLMClient(empty_extraction_responder) if args.provider == "fake" else make_client(settings)
        model = client.model
        gateway = LLMGateway(client, temperature=settings.temperature)
        sentences = [Sentence(id=sid, review_id=sid, text=t) for sid, t in texts.items()]
        examples = load_fewshot(Path(args.fewshot)) if args.fewshot else []
        res = extract_mentions(gateway, sentences, taxonomy, examples=examples, batch_size=args.batch_size)
        pred = mentions_to_labels(res.mentions, taxonomy)
        # Предложения без упоминаний тоже считаются ответом модели
        for sid in texts:
            if sid not in res.failed_sentence_ids:
                pred.setdefault(sid, {})
        summary = gateway.summary()
        errors = res.errors
    elapsed = time.perf_counter() - started

    metrics = evaluate(gold, pred)
    metrics.update({
        "provider": args.provider,
        "model": model,
        "split": args.split,
        "taxonomy": taxonomy.version,
        "elapsed_sec": round(elapsed, 2),
        "llm": summary,
        "failed_batches": len(errors),
    })

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_model = re.sub(r"[^\w.-]+", "-", model)
    out = ROOT / "runs" / "eval" / f"{stamp}_{args.split}_{args.provider}_{safe_model}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.json").write_text(json.dumps(pred, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.provider != "baseline":
        with (out / "llm_calls.jsonl").open("w", encoding="utf-8") as f:
            for r in gateway.records:
                f.write(r.model_dump_json() + "\n")
        (out / "errors.txt").write_text("\n".join(errors), encoding="utf-8")

    a, s, st = metrics["aspects"], metrics["sentiment"], metrics["strict"]
    print(f"{args.provider} / {model}, выборка {args.split}: {metrics['sentences']} предложений")
    print(f"  аспекты:     P={a['precision']:.3f} R={a['recall']:.3f} F1={a['f1']:.3f}")
    print(f"  тональность: accuracy={s['accuracy']:.3f} macro-F1={s['macro_f1']:.3f} (пар: {s['pairs']})")
    print(f"  строгий F1:  {st['f1']:.3f}")
    if summary:
        tot = {k: sum(v[k] for v in summary.values()) for k in ("calls", "tokens_in", "tokens_out", "cost", "errors")}
        print(f"  запросов {tot['calls']:.0f}, токенов {tot['tokens_in']:.0f}+{tot['tokens_out']:.0f}, "
              f"стоимость {tot['cost']:.2f} ₽, ответов не по формату {tot['errors']:.0f}, неудачных пакетов {len(errors)}")
    print(f"  время {elapsed:.1f} с → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
