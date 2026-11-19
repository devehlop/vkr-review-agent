"""Чтение размеченных выборок из data/annotation/ (CSV или XLSX)."""

from __future__ import annotations

import csv
from pathlib import Path

from review_agent.taxonomy import Taxonomy, load_taxonomy

# Разметка: id предложения → {код атрибута → тональность}
Labels = dict[str, dict[str, int]]


def parse_sentiment(value: object) -> int | None:
    """Приводит значение ячейки к −1, 0, 1 или None (пусто)."""
    if value is None:
        return None
    s = str(value).strip().replace("−", "-").replace(",", ".")
    if s == "":
        return None
    try:
        v = int(float(s))
    except ValueError as e:
        raise ValueError(f"Недопустимое значение тональности: {value!r}") from e
    if v not in (-1, 0, 1):
        raise ValueError(f"Недопустимое значение тональности: {value!r}")
    return v


def _rows_from_xlsx(path: Path) -> list[dict[str, object]]:
    from openpyxl import load_workbook

    ws = load_workbook(path, read_only=True, data_only=True).active
    rows = list(ws.iter_rows(values_only=True))
    # В файлах для разметки первая строка — русские названия, вторая — коды колонок
    # (в первой строке у служебных колонок тоже стоит «id»), поэтому берём последнюю
    header_idx = max(i for i, r in enumerate(rows[:3]) if r and r[0] == "id")
    header = [str(h) if h is not None else "" for h in rows[header_idx]]
    return [dict(zip(header, r)) for r in rows[header_idx + 1 :] if r and r[0]]


def _rows_from_csv(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [dict(r) for r in csv.DictReader(f) if r.get("id")]


def load_annotation(path: str | Path, taxonomy: Taxonomy | None = None) -> Labels:
    """Загружает разметку. Предложение без отметок даёт пустой словарь."""
    path = Path(path)
    taxonomy = taxonomy or load_taxonomy()
    codes = [a.code for a in taxonomy.attributes]
    rows = _rows_from_xlsx(path) if path.suffix.lower() == ".xlsx" else _rows_from_csv(path)
    labels: Labels = {}
    for r in rows:
        sid = str(r["id"])
        marks: dict[str, int] = {}
        for code in codes:
            try:
                v = parse_sentiment(r.get(code))
            except ValueError as e:
                raise ValueError(f"{path.name}, {sid}, колонка {code}: {e}") from e
            if v is not None:
                marks[code] = v
        labels[sid] = marks
    return labels


def load_sentences(path: str | Path) -> dict[str, str]:
    """id → текст предложения (для передачи в модель)."""
    path = Path(path)
    rows = _rows_from_xlsx(path) if path.suffix.lower() == ".xlsx" else _rows_from_csv(path)
    return {str(r["id"]): str(r["sentence"]) for r in rows}
