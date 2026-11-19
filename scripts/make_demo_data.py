"""Демонстрационный набор «свой товар + конкурент» из RuReviews.

Тексты отзывов реальные (очищенный RuReviews без персональных данных),
а разбивка по товарам, оценки и даты сгенерированы: в RuReviews нет
привязки к товару. Набор нужен только для демонстрации и отладки,
выводы по нему ничего не говорят о реальных товарах.

У «своего» товара чаще встречаются жалобы на размер, у «конкурента» —
на ткань, чтобы в отчёте было видно сравнение.

Запуск: python scripts/make_demo_data.py  (сначала prepare_rureviews.py)
"""

from __future__ import annotations

import csv
import random
import re
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from prepare_rureviews import FILE, RAW_DIR, download, load_reviews  # noqa: E402

OUT = ROOT / "data" / "samples"
SEED = 7
SIZE_RE = re.compile(r"маломер|большемер|размер", re.IGNORECASE)
FABRIC_RE = re.compile(r"ткан|синтет|тонк|материал", re.IGNORECASE)
RATING = {"negative": (1, 2), "neutral": (3, 3), "positive": (4, 5)}


def pick(rng, pool, n):
    items = rng.sample(pool, min(n, len(pool)))
    for r in items:
        pool.remove(r)
    return items


def write(path: Path, rows: list[dict], rng: random.Random) -> None:
    start = date(2026, 5, 1)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Номер отзыва", "Дата публикации", "Оценка", "Комментарий"])
        for k, r in enumerate(rows, start=1):
            created = start + timedelta(days=rng.randrange(180))
            w.writerow([k, created.strftime("%d.%m.%Y"), rng.randint(*RATING[r["label"]]), r["text"]])


def main() -> None:
    raw = RAW_DIR / FILE
    download(raw)
    reviews, _ = load_reviews(raw)
    # Отзывы из выборок для разметки в демо не берём
    used = set()
    for split in ("fewshot", "mini", "test"):
        with (ROOT / "data" / "annotation" / f"{split}.csv").open(encoding="utf-8") as f:
            used |= {r["review_id"] for r in csv.DictReader(f)}
    pool = [r for r in reviews if r["review_id"] not in used and len(r["text"]) <= 400]
    rng = random.Random(SEED)

    size_neg = [r for r in pool if r["label"] == "negative" and SIZE_RE.search(r["text"])]
    fabric_neg = [r for r in pool if r["label"] == "negative" and FABRIC_RE.search(r["text"]) and not SIZE_RE.search(r["text"])]
    positive = [r for r in pool if r["label"] == "positive"]
    neutral = [r for r in pool if r["label"] == "neutral"]

    own = pick(rng, size_neg, 45) + pick(rng, fabric_neg, 10) + pick(rng, positive, 95) + pick(rng, neutral, 30)
    comp = pick(rng, size_neg, 8) + pick(rng, fabric_neg, 35) + pick(rng, positive, 80) + pick(rng, neutral, 25)
    rng.shuffle(own)
    rng.shuffle(comp)
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / "demo_own.csv", own, rng)
    write(OUT / "demo_competitor.csv", comp, rng)
    print(f"свой товар: {len(own)}, конкурент: {len(comp)} → data/samples/")


if __name__ == "__main__":
    main()
