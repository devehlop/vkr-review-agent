"""Подготовка датасета RuReviews и выборок для ручной разметки.

Что делает:
1. Скачивает RuReviews (Smetanin, Komarov, 2019; лицензия Apache 2.0) в data/raw/
   по зафиксированному коммиту и проверяет контрольную сумму.
2. Выбрасывает отзывы, где найдены персональные данные (телефоны, почта,
   ссылки, номера заказов). Сырые данные в репозиторий не попадают.
3. Делит отзывы на предложения и делает три непересекающиеся выборки
   (по одному предложению из отзыва, поровну по классам тональности отзыва):
   - fewshot  — 30 предложений для примеров в промпте;
   - mini     — 100 предложений для мини-эксперимента по выбору LLM;
   - test     — 300 предложений для итоговой оценки качества.
4. Сохраняет выборки в data/annotation/ в CSV и XLSX (с выпадающими списками).

Запуск из корня репозитория:
    python scripts/prepare_rureviews.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import unicodedata
import urllib.request
from pathlib import Path

from razdel import sentenize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from review_agent.preprocess.pii import есть_пд as has_pii  # noqa: E402

# Справочник для эталонной разметки: фиксированный, чтобы все модели в
# мини-эксперименте сравнивались по одним и тем же атрибутам
TAXONOMY_PATH = ROOT / "eval" / "taxonomy_clothing_v1.json"

# Очистка и разбиение на предложения взяты из кода PR #2 без изменений:
# с ними выборки в data/annotation/ воспроизводятся байт в байт.
_SPACES = re.compile(r"[ \t\u00a0\u200b]+")
_MANY_NEWLINES = re.compile(r"\n{2,}")
_REPEATED_PUNCT = re.compile(r"([!?.,)(])\1{3,}")
_CONTROL = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]")


def clean_text(text: str | None) -> str:
    """Нормализует текст: юникод, служебные символы, пробелы, повторы знаков."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    text = _REPEATED_PUNCT.sub(r"\1\1\1", text)
    text = _SPACES.sub(" ", text)
    text = _MANY_NEWLINES.sub("\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def normalize_for_compare(text: str) -> str:
    """Ключ для поиска дублей: нижний регистр, ё → е, только буквы и цифры."""
    text = text.lower().replace("ё", "е")
    return " ".join(re.findall(r"\w+", text))


def split_sentences(text: str) -> list[str]:
    """Предложения razdel; перевод строки тоже граница."""
    result: list[str] = []
    for line in text.split("\n"):
        for sent in sentenize(line):
            s = sent.text.strip()
            if s and re.search(r"[^\W\d_]", s):
                result.append(s)
    return result


class _Attr:
    def __init__(self, d: dict):
        self.code, self.attribute = d["code"], d["attribute"]


class _Taxonomy:
    def __init__(self, d: dict):
        self.attributes = [_Attr(a) for a in d["attributes"]]


def load_taxonomy() -> _Taxonomy:
    return _Taxonomy(json.loads(TAXONOMY_PATH.read_text(encoding="utf-8")))

COMMIT = "2f6e61d30dc6c58ba47dce72e5803e1f0a3f38a7"
FILE = "women-clothing-accessories.3-class.balanced.csv"
URL = f"https://raw.githubusercontent.com/sismetanin/rureviews/{COMMIT}/{FILE}"
SHA256 = "db226708d966a1506560136ef85b33ea326eaf6119465d4b1edb5bf9395330fd"

RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "annotation"

# В датасете класс нейтральных отзывов записан с опечаткой
LABELS = {"negative": "negative", "neautral": "neutral", "neutral": "neutral", "positive": "positive"}

SPLITS = [("fewshot", 30), ("mini", 100), ("test", 300)]
SEED = 2026
MIN_WORDS, MAX_WORDS = 3, 40


def download(path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Скачиваю {URL}")
        urllib.request.urlretrieve(URL, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"Контрольная сумма не совпала: {digest}")


def load_reviews(path: Path) -> tuple[list[dict], dict[str, int]]:
    stats = {"всего": 0, "с персональными данными": 0, "пустые или дубли": 0}
    seen: set[str] = set()
    reviews: list[dict] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            stats["всего"] += 1
            text = clean_text(row["review"])
            if has_pii(text):
                stats["с персональными данными"] += 1
                continue
            key = normalize_for_compare(text)
            if not key or key in seen:
                stats["пустые или дубли"] += 1
                continue
            seen.add(key)
            reviews.append({"review_id": f"rr{i:05d}", "label": LABELS[row["sentiment"]], "text": text})
    stats["осталось"] = len(reviews)
    return reviews, stats


def make_splits(reviews: list[dict]) -> dict[str, list[dict]]:
    rng = random.Random(SEED)
    by_label: dict[str, list[dict]] = {"negative": [], "neutral": [], "positive": []}
    for r in reviews:
        by_label[r["label"]].append(r)
    for items in by_label.values():
        rng.shuffle(items)

    iters = {label: iter(items) for label, items in by_label.items()}
    labels = list(by_label)

    def next_sentence(label: str) -> dict:
        # Берём следующий отзыв класса и случайное подходящее предложение из него
        for r in iters[label]:
            sents = [s for s in split_sentences(r["text"]) if MIN_WORDS <= len(s.split()) <= MAX_WORDS]
            if sents:
                s = rng.choice(sents)
                return {**r, "sentence": s}
        raise RuntimeError(f"Не хватило отзывов класса {label}")

    result: dict[str, list[dict]] = {}
    for name, size in SPLITS:
        rows = [next_sentence(labels[k % 3]) for k in range(size)]
        rng.shuffle(rows)
        result[name] = [{"id": f"{name}-{k + 1:03d}", **row} for k, row in enumerate(rows)]
    return result


def save_split(name: str, rows: list[dict]) -> None:
    taxonomy = load_taxonomy()
    codes = [a.code for a in taxonomy.attributes]
    header = ["id", "review_id", "review_label", "sentence", *codes, "features", "note", "review_text"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with (OUT_DIR / f"{name}.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r["id"], r["review_id"], r["label"], r["sentence"], *[""] * len(codes), "", "", r["text"]])

    save_xlsx(name, rows, header, taxonomy)


def save_xlsx(name: str, rows: list[dict], header: list[str], taxonomy) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    codes = [a.code for a in taxonomy.attributes]
    wb = Workbook()
    ws = wb.active
    ws.title = "разметка"
    # Над кодами атрибутов показываем их русские названия
    titles = {a.code: a.attribute for a in taxonomy.attributes}
    ws.append([titles.get(h, h) for h in header])
    ws.append(header)
    for r in rows:
        ws.append([r["id"], r["review_id"], r["label"], r["sentence"], *[None] * len(codes), None, None, r["text"]])

    bold = Font(bold=True)
    fill = PatternFill("solid", fgColor="FFF2CC")
    for row in ws.iter_rows(min_row=1, max_row=2):
        for c in row:
            c.font = bold
            c.alignment = Alignment(wrap_text=True, vertical="top")
    first, last = 5, 4 + len(codes)
    dv = DataValidation(type="list", formula1='"-1,0,1"', allow_blank=True, showErrorMessage=True,
                        errorTitle="Тональность", error="Допустимо: -1, 0, 1 или пусто")
    ws.add_data_validation(dv)
    dv.add(f"{get_column_letter(first)}3:{get_column_letter(last)}{len(rows) + 2}")
    for col in range(first, last + 1):
        ws.column_dimensions[get_column_letter(col)].width = 9
        for row in range(3, len(rows) + 3):
            ws.cell(row=row, column=col).fill = fill
    widths = {"A": 11, "B": 9, "C": 9, "D": 60}
    for col, w in widths.items():
        ws.column_dimensions[col].width = w
    ws.column_dimensions[get_column_letter(last + 1)].width = 30
    ws.column_dimensions[get_column_letter(last + 2)].width = 20
    ws.column_dimensions[get_column_letter(last + 3)].width = 80
    for row in ws.iter_rows(min_row=3, min_col=4, max_col=4):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "E3"
    wb.save(OUT_DIR / f"{name}.xlsx")


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    raw = RAW_DIR / FILE
    download(raw)
    reviews, stats = load_reviews(raw)
    for k, v in stats.items():
        print(f"{k}: {v}")
    for name, rows in make_splits(reviews).items():
        save_split(name, rows)
        print(f"{name}: {len(rows)} предложений → data/annotation/{name}.csv, .xlsx")


if __name__ == "__main__":
    main()
