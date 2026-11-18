"""Мини-эксперимент: сравнение моделей на размеченной выборке (шаг 4 конвейера).

Запуск:
    python -m eval.run_experiment --models gigachat yandex qwen

Для каждой модели считаются: F1 по аспектам, macro-F1 по тональности, доля ответов
с нарушением формата, время и стоимость. Итог: таблица в eval/results/.
"""
import argparse
import json
from datetime import date
from pathlib import Path

from eval.metrics import aspect_metrics
from eval.sample import read_sample
from review_agent.config import MODEL_NAMES, make_client
from review_agent.llm.base import LLMClient
from review_agent.llm.gateway import LLMGateway
from review_agent.models import Sentence
from review_agent.pipeline.extract import PROMPT_VERSION, extract_mentions

ROOT = Path(__file__).resolve().parents[1]


def run_model(client: LLMClient, sentences: list[Sentence], gold: dict, taxonomy: dict[str, str],
              batch_size: int = 20, journal_dir: Path | None = None, cache_dir: Path | None = None) -> dict:
    """Прогоняет одну модель по выборке и возвращает метрики и расход."""
    gateway = LLMGateway(
        client,
        journal_path=journal_dir / f"{client.name}.jsonl" if journal_dir else None,
        cache_dir=cache_dir / client.name if cache_dir else None,
    )
    result = extract_mentions(gateway, sentences, taxonomy, batch_size=batch_size)
    predicted = {}
    for sentence_id, mentions in result.mentions.items():
        labels = {}
        for mention in mentions:
            labels.setdefault(mention.attribute, mention.sentiment)  # повтор атрибута: берём первое упоминание
        predicted[sentence_id] = labels
    usage = gateway.usage
    requests = usage.calls + usage.cached
    return {
        "модель": client.name,
        "название_модели": client.model,
        "промпт": PROMPT_VERSION,
        **aspect_metrics(gold, predicted),
        "запросов": requests,
        "ответов_с_ошибкой_формата": usage.format_errors,
        "доля_ошибок_формата": round(usage.format_errors / requests, 3) if requests else 0.0,
        "предложений_без_ответа": len(result.failed_ids),
        "токенов_ввод": usage.tokens_in,
        "токенов_вывод": usage.tokens_out,
        "стоимость": round(usage.cost, 2),
        "секунд": round(usage.seconds, 1),
    }


def summary_table(rows: list[dict]) -> str:
    """Таблица Markdown для отчёта и главы 3."""
    header = ["Модель", "F1 аспекты", "Precision", "Recall", "Тональность macro-F1", "Тональность accuracy",
              "Ошибки формата", "Без ответа", "Токены (ввод/вывод)", "Стоимость", "Время, с"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in [
            r["модель"], r["f1"], r["precision"], r["recall"], r["тональность_macro_f1"],
            r["тональность_accuracy"], r["доля_ошибок_формата"], r["предложений_без_ответа"],
            f'{r["токенов_ввод"]}/{r["токенов_вывод"]}', r["стоимость"], r["секунд"]]) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    parser.add_argument("--sample", default=str(ROOT / "data/eval/sample_100.csv"))
    parser.add_argument("--taxonomy", default=str(ROOT / "eval/taxonomy_clothing.json"))
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--out", default=str(ROOT / "eval/results"))
    args = parser.parse_args()

    taxonomy = json.loads(Path(args.taxonomy).read_text(encoding="utf-8"))
    sentences, gold = read_sample(args.sample, list(taxonomy))
    unlabeled = [i for i, labels in gold.items() if labels is None]
    if unlabeled:
        raise SystemExit(
            f"Не размечено предложений: {len(unlabeled)} из {len(gold)} (первые номера: {unlabeled[:10]}). "
            "Заполните колонку «разметка» в файле выборки, см. eval/README.md."
        )

    rows = []
    for name in args.models:
        print(f"Модель {name}: запуск на {len(sentences)} предложениях…")
        row = run_model(make_client(name), sentences, gold, taxonomy, args.batch_size,
                        journal_dir=ROOT / "logs", cache_dir=ROOT / ".cache")
        rows.append(row)
        print(f"  F1 аспекты {row['f1']}, тональность macro-F1 {row['тональность_macro_f1']}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    (out / f"{stamp}_experiment.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    table = summary_table(rows)
    (out / f"{stamp}_experiment.md").write_text(
        f"# Сравнение моделей, {stamp}\n\nВыборка: {len(sentences)} предложений, промпт {PROMPT_VERSION}.\n\n{table}\n",
        encoding="utf-8")
    print("\n" + table)


if __name__ == "__main__":
    main()
