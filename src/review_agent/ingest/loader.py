"""Загрузка отзывов из CSV/XLSX с сопоставлением колонок (FR-1).

Читаются только сопоставленные колонки. Всё остальное (имя автора, фото, ссылки
на профиль) отбрасывается при чтении: персональные данные не обрабатываются
и не хранятся (docs/tz.md).
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

# Поля отзыва, которые умеем читать, и типичные названия колонок в выгрузках.
СИНОНИМЫ: dict[str, tuple[str, ...]] = {
    "text": ("text", "текст", "отзыв", "комментарий", "comment", "review", "content"),
    "pros": ("pros", "достоинства", "плюсы", "advantages"),
    "cons": ("cons", "недостатки", "минусы", "disadvantages"),
    "rating": ("rating", "оценка", "рейтинг", "stars", "звёзды", "звезды", "productvaluation"),
    "date": ("date", "дата", "created", "createdat", "дата отзыва"),
    "external_id": ("id", "review_id", "идентификатор"),
}


class ЗагрузкаОшибка(ValueError):
    """Файл нельзя прочитать или в нём нет нужных колонок."""


@dataclass
class Отзыв:
    """Один отзыв без персональных данных."""
    id: str                # внешний id или номер строки
    product: str           # название товара или метка вроде «свой» / «конкурент А»
    is_own: bool           # свой товар или конкурент
    text: str = ""
    pros: str = ""
    cons: str = ""
    rating: float | None = None
    date: date | None = None


def _нормализовать(имя: str) -> str:
    return "".join(ch for ch in str(имя).lower().strip() if ch.isalnum() or ch == " ").replace(" ", "")


def определить_колонки(заголовки: list[str]) -> dict[str, str]:
    """Автоматически сопоставляет поля отзыва с колонками файла по названиям."""
    найдено: dict[str, str] = {}
    нормализованные = {_нормализовать(h): h for h in заголовки if h}
    for поле, варианты in СИНОНИМЫ.items():
        for вариант in варианты:
            ключ = _нормализовать(вариант)
            if ключ in нормализованные:
                найдено[поле] = нормализованные[ключ]
                break
    return найдено


def _читать_csv(путь: Path) -> list[dict]:
    сырые = путь.read_bytes()
    for кодировка in ("utf-8-sig", "cp1251"):
        try:
            текст = сырые.decode(кодировка)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ЗагрузкаОшибка(f"Не удалось определить кодировку файла {путь.name}")
    try:
        диалект = csv.Sniffer().sniff(текст[:4096], delimiters=",;\t")
    except csv.Error:
        диалект = csv.excel
    return list(csv.DictReader(io.StringIO(текст), dialect=диалект))


def _читать_xlsx(путь: Path, лист: str | None) -> list[dict]:
    import openpyxl
    книга = openpyxl.load_workbook(путь, read_only=True, data_only=True)
    try:
        ws = книга[лист] if лист else книга.active
        строки = list(ws.iter_rows(values_only=True))
    finally:
        книга.close()
    if not строки:
        return []
    заголовки = [str(h) if h is not None else "" for h in строки[0]]
    return [dict(zip(заголовки, r)) for r in строки[1:]]


def _в_текст(значение) -> str:
    return "" if значение is None else str(значение).strip()


def _в_число(значение) -> float | None:
    try:
        return float(str(значение).replace(",", ".").strip())
    except (TypeError, ValueError):
        return None


def _в_дату(значение) -> date | None:
    if isinstance(значение, datetime):
        return значение.date()
    if isinstance(значение, date):
        return значение
    s = _в_текст(значение)
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    try:
        return datetime.strptime(s[:10], "%d.%m.%Y").date()
    except ValueError:
        return None


def загрузить(путь: str | Path, *, product: str, is_own: bool = True,
              колонки: dict[str, str] | None = None, лист: str | None = None) -> list[Отзыв]:
    """Читает файл и возвращает отзывы.

    колонки: ручное сопоставление {поле: название колонки}; недостающее
    определяется автоматически. Нужна хотя бы одна из колонок text/pros/cons.
    """
    путь = Path(путь)
    расширение = путь.suffix.lower()
    if расширение == ".csv":
        строки = _читать_csv(путь)
    elif расширение in (".xlsx", ".xlsm"):
        строки = _читать_xlsx(путь, лист)
    else:
        raise ЗагрузкаОшибка(f"Неподдерживаемый формат {расширение}: нужен CSV или XLSX")
    if not строки:
        return []

    сопоставление = определить_колонки(list(строки[0].keys()))
    сопоставление.update(колонки or {})
    if not any(п in сопоставление for п in ("text", "pros", "cons")):
        raise ЗагрузкаОшибка(
            "Не найдена колонка с текстом отзыва. Укажите колонки вручную, "
            f"доступные в файле: {list(строки[0].keys())}")

    def взять(строка: dict, поле: str):
        имя = сопоставление.get(поле)
        return строка.get(имя) if имя else None

    результат = []
    for номер, строка in enumerate(строки, start=1):
        внешний = _в_текст(взять(строка, "external_id"))
        результат.append(Отзыв(
            id=внешний or f"{путь.stem}:{номер}",
            product=product, is_own=is_own,
            text=_в_текст(взять(строка, "text")),
            pros=_в_текст(взять(строка, "pros")),
            cons=_в_текст(взять(строка, "cons")),
            rating=_в_число(взять(строка, "rating")),
            date=_в_дату(взять(строка, "date")),
        ))
    return результат
