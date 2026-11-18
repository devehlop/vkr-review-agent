"""Чтение и подготовка выборки предложений для разметки.

Формат файла выборки (CSV, разделитель «;», открывается в Excel):
id; предложение; тональность_отзыва; разметка

Колонка «разметка» заполняется вручную:
  Размер и посадка=-1; Ткань и материал=1   — упоминания атрибутов с тональностью (-1, 0, 1)
  -                                         — в предложении нет упоминаний
  (пусто)                                   — предложение ещё не размечено
"""
import csv
import random
from pathlib import Path

from review_agent.models import Sentence
from review_agent.preprocess import normalize_text
from review_agent.preprocess.sentences import split_text

COLUMNS = ["id", "предложение", "тональность_отзыва", "разметка"]


def build_sample(reviews: list[tuple[str, str]], size: int = 100, seed: int = 42,
                 min_chars: int = 20, max_chars: int = 250) -> list[dict]:
    """Случайная выборка предложений, поровну из отзывов каждого класса тональности.

    reviews: пары (текст отзыва, метка тональности отзыва).
    """
    by_label: dict[str, list[str]] = {}
    for text, label in reviews:
        for sentence in split_text(normalize_text(text)):
            if min_chars <= len(sentence) <= max_chars:
                by_label.setdefault(label, []).append(sentence)
    rng = random.Random(seed)
    labels = sorted(by_label)
    rows, used = [], set()
    for index, label in enumerate(labels):
        # остаток от деления раздаём первым классам, чтобы в сумме получилось ровно size
        quota = size // len(labels) + (1 if index < size % len(labels) else 0)
        pool = [s for s in dict.fromkeys(by_label[label]) if s not in used]
        chosen = rng.sample(pool, min(quota, len(pool)))
        used.update(chosen)
        rows.extend({"предложение": s, "тональность_отзыва": label, "разметка": ""} for s in chosen)
    rng.shuffle(rows)
    for number, row in enumerate(rows, start=1):
        row["id"] = number
    return rows


def write_sample(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def parse_labels(cell: str, attributes: list[str], row_id: int) -> dict[str, int] | None:
    """Разбирает ячейку разметки. None: не размечено. Пустой словарь: упоминаний нет."""
    cell = (cell or "").strip()
    if not cell:
        return None
    if cell in ("-", "–", "—", "нет"):
        return {}
    labels = {}
    for part in cell.split(";"):
        if not part.strip():
            continue
        name, _, value = part.rpartition("=")
        name, value = name.strip(), value.strip().replace("−", "-").replace("+", "")
        if name not in attributes:
            raise ValueError(f"Строка {row_id}: атрибута «{name}» нет в справочнике")
        if value not in ("-1", "0", "1"):
            raise ValueError(f"Строка {row_id}: тональность «{value}» должна быть -1, 0 или 1")
        labels[name] = int(value)
    return labels


def read_sample(path: str | Path, attributes: list[str]) -> tuple[list[Sentence], dict[int, dict[str, int] | None]]:
    """Возвращает предложения и эталонную разметку (номер предложения -> атрибут -> тональность)."""
    sentences, gold = [], {}
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file, delimiter=";"):
            row_id = int(row["id"])
            sentences.append(Sentence(row_id, row_id, row["предложение"], "комментарий"))
            gold[row_id] = parse_labels(row.get("разметка", ""), attributes, row_id)
    return sentences, gold
