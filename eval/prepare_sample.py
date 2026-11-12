"""Готовит выборку предложений из RuReviews для ручной разметки.

Запуск:
    python -m eval.prepare_sample data/raw/women-clothing-accessories.3-class.balanced.csv

Файл датасета скачать из https://github.com/sismetanin/rureviews (лицензия Apache-2.0).
"""
import argparse
import csv

from eval.sample import build_sample, write_sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="файл RuReviews: колонки review и sentiment, разделитель табуляция")
    parser.add_argument("--out", default="data/eval/sample_100.csv")
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.source, encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
        # строки без метки тональности (обрыв формата в исходном файле) пропускаем
        reviews = [(row["review"], row["sentiment"]) for row in reader
                   if row.get("review") and row.get("sentiment") in ("negative", "neautral", "neutral", "positive")]
    rows = build_sample(reviews, size=args.size, seed=args.seed)
    write_sample(rows, args.out)
    print(f"Отзывов прочитано: {len(reviews)}. Предложений в выборке: {len(rows)}. Файл: {args.out}")


if __name__ == "__main__":
    main()
