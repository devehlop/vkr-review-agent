"""Загрузка отзывов из CSV/XLSX с сопоставлением колонок (FR-1)."""
from pathlib import Path

import pandas as pd

from review_agent.models import Review

# Стандартное поле -> возможные названия колонки в файле (в нижнем регистре)
SYNONYMS = {
    "text": ["text", "review", "отзыв", "текст", "текст отзыва", "комментарий"],
    "rating": ["rating", "оценка", "рейтинг", "звёзды", "звезды", "stars"],
    "created": ["date", "created", "дата", "дата отзыва", "дата публикации"],
    "pros": ["pros", "достоинства", "плюсы"],
    "cons": ["cons", "недостатки", "минусы"],
    "variant": ["variant", "вариант", "размер", "цвет"],
}


def guess_mapping(columns: list[str]) -> dict[str, str]:
    """Подбирает соответствие «стандартное поле -> колонка файла» по названиям."""
    lower = {str(c).strip().lower(): c for c in columns}
    mapping = {}
    for field_name, names in SYNONYMS.items():
        for name in names:
            if name in lower:
                mapping[field_name] = lower[name]
                break
    return mapping


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, dtype=str)
    # sep=None: pandas сам определяет разделитель (запятая, точка с запятой, табуляция)
    return pd.read_csv(path, dtype=str, sep=None, engine="python", encoding="utf-8-sig")


def _cell(row, column) -> str:
    if column is None:
        return ""
    value = row.get(column)
    return "" if value is None or pd.isna(value) else str(value).strip()


def load_reviews(
    path: str | Path,
    product: str,
    mapping: dict[str, str] | None = None,
    is_own: bool = True,
) -> list[Review]:
    """Читает файл и возвращает список отзывов.

    mapping задаёт соответствие колонок вручную, иначе оно подбирается по названиям.
    Читаются только сопоставленные колонки: остальные (имя покупателя, фото) игнорируются.
    """
    path = Path(path)
    table = _read_table(path)
    mapping = mapping or guess_mapping(list(table.columns))
    if not any(k in mapping for k in ("text", "pros", "cons")):
        raise ValueError(
            f"В файле {path.name} не найдена колонка с текстом отзыва. "
            f"Колонки файла: {list(table.columns)}. Укажите соответствие в mapping."
        )

    reviews = []
    for number, (_, row) in enumerate(table.iterrows(), start=1):
        rating_raw = _cell(row, mapping.get("rating"))
        try:
            rating = int(float(rating_raw.replace(",", "."))) if rating_raw else None
        except ValueError:
            rating = None
        created_raw = _cell(row, mapping.get("created"))
        created = pd.to_datetime(created_raw, errors="coerce", dayfirst=True) if created_raw else pd.NaT
        reviews.append(
            Review(
                id=number,
                product=product,
                text=_cell(row, mapping.get("text")),
                rating=rating,
                created=None if pd.isna(created) else created.date(),
                pros=_cell(row, mapping.get("pros")),
                cons=_cell(row, mapping.get("cons")),
                variant=_cell(row, mapping.get("variant")),
                is_own=is_own,
            )
        )
    return reviews
