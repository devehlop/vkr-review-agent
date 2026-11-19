"""Загрузка отзывов из CSV/XLSX с сопоставлением колонок (FR-1).

Выгрузки разных площадок называют колонки по-разному, поэтому колонки
сопоставляются по словарю синонимов, а продавец может поправить результат
в интерфейсе. Ошибочные строки не роняют загрузку, а попадают в список
пропущенных с причиной (NFR-6).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ValidationError

from review_agent.models import Review

# Поле модели → варианты названий колонок (сравниваются без регистра и знаков)
SYNONYMS: dict[str, list[str]] = {
    "text": ["text", "review", "comment", "текст", "отзыв", "текст отзыва", "комментарий"],
    "pros": ["pros", "advantages", "достоинства", "плюсы"],
    "cons": ["cons", "disadvantages", "недостатки", "минусы"],
    "rating": ["rating", "score", "stars", "оценка", "рейтинг", "звезды", "количество звезд"],
    "created": ["date", "created", "created_at", "дата", "дата отзыва", "дата создания", "дата публикации"],
    "variant": ["variant", "size", "color", "вариант", "размер", "цвет", "характеристики"],
    "id": ["id", "review_id", "ид", "id отзыва", "номер отзыва"],
}


class ColumnMapping(BaseModel):
    """Какая колонка файла соответствует какому полю отзыва. None — колонки нет."""

    text: str | None = None
    pros: str | None = None
    cons: str | None = None
    rating: str | None = None
    created: str | None = None
    variant: str | None = None
    id: str | None = None

    def is_usable(self) -> bool:
        return any([self.text, self.pros, self.cons])


@dataclass
class LoadResult:
    reviews: list[Review]
    mapping: ColumnMapping
    skipped: list[tuple[int, str]] = field(default_factory=list)


def _key(name: str) -> str:
    return " ".join(re.findall(r"\w+", str(name).lower().replace("ё", "е")))


def guess_mapping(columns: list[str]) -> ColumnMapping:
    """Подбирает колонки по синонимам: сначала точное совпадение, потом вхождение."""
    keys = {c: _key(c) for c in columns}
    used: set[str] = set()
    found: dict[str, str] = {}
    for exact in (True, False):
        for fld, names in SYNONYMS.items():
            if fld in found:
                continue
            for col, k in keys.items():
                if col in used:
                    continue
                hit = k in names if exact else any(n in k for n in names if len(n) > 3)
                if hit:
                    found[fld] = col
                    used.add(col)
                    break
    return ColumnMapping(**found)


def read_table(path: str | Path) -> pd.DataFrame:
    """Читает CSV (разделитель определяется автоматически) или XLSX как строки."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str)
    return pd.read_csv(path, sep=None, engine="python", dtype=str, encoding="utf-8-sig")


def _value(row: pd.Series, col: str | None) -> str:
    if not col:
        return ""
    v = row.get(col)
    return "" if v is None or pd.isna(v) else str(v).strip()


def parse_rating(value: str) -> int | None:
    if not value:
        return None
    m = re.search(r"\d+(?:[.,]\d+)?", value)
    if not m:
        raise ValueError(f"оценка не распознана: {value!r}")
    rating = round(float(m.group(0).replace(",", ".")))
    if not 1 <= rating <= 5:
        raise ValueError(f"оценка вне диапазона 1–5: {value!r}")
    return rating


def parse_date(value: str) -> date | None:
    if not value:
        return None
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d.%m.%Y %H:%M", "%d.%m.%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value[:19], fmt).date()
        except ValueError:
            continue
    try:
        return pd.to_datetime(value, dayfirst=True).date()
    except (ValueError, TypeError) as e:
        raise ValueError(f"дата не распознана: {value!r}") from e


def load_reviews(
    path: str | Path,
    product_id: str,
    mapping: ColumnMapping | None = None,
) -> LoadResult:
    """Загружает отзывы одного товара. Пустые строки пропускаются с причиной."""
    df = read_table(path)
    mapping = mapping or guess_mapping(list(df.columns))
    if not mapping.is_usable():
        raise ValueError(f"Не найдена колонка с текстом отзыва. Колонки файла: {list(df.columns)}")

    reviews: list[Review] = []
    skipped: list[tuple[int, str]] = []
    for i, row in df.iterrows():
        line = int(i) + 2  # номер строки в файле с учётом заголовка
        text, pros, cons = _value(row, mapping.text), _value(row, mapping.pros), _value(row, mapping.cons)
        if not (text or pros or cons):
            skipped.append((line, "пустой отзыв"))
            continue
        try:
            review = Review(
                id=f"{product_id}:{_value(row, mapping.id) or line}",
                product_id=product_id,
                text=text,
                pros=pros,
                cons=cons,
                rating=parse_rating(_value(row, mapping.rating)),
                created=parse_date(_value(row, mapping.created)),
                variant=_value(row, mapping.variant),
            )
        except (ValueError, ValidationError) as e:
            skipped.append((line, str(e).splitlines()[0]))
            continue
        reviews.append(review)
    return LoadResult(reviews=reviews, mapping=mapping, skipped=skipped)
