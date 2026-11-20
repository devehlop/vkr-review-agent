"""Предобработка отзывов (FR-2…FR-4, шаг 2 из docs/architecture.md).

1. Нормализация текста и удаление пустых отзывов и точных дубликатов.
2. Разбиение на предложения (razdel) с сохранением поля-источника
   (text / pros / cons): предложение из «Недостатков» почти всегда негативное.
3. Метка подозрительных отзывов: почти одинаковые тексты у разных отзывов.
   Они не удаляются, а только помечаются.

Для меток используется сходство по словесным шинглам. Это упрощённая замена
сходству эмбеддингов из архитектуры; её можно заменить, когда появятся эмбеддинги.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from razdel import sentenize

from review_agent.ingest.loader import Отзыв

ПОЛЯ = ("pros", "cons", "text")          # порядок, в котором склеиваем отзыв
_НЕВИДИМЫЕ = re.compile(r"[​-‏⁠﻿]")
_УПРАВЛЯЮЩИЕ = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass
class Предложение:
    review_id: str
    idx: int                 # номер предложения внутри отзыва
    text: str
    source: str             # text / pros / cons
    product: str
    is_own: bool


@dataclass
class Результат:
    reviews: list[Отзыв]
    sentences: list[Предложение]
    suspicious_ids: set[str] = field(default_factory=set)
    stats: dict = field(default_factory=dict)


def нормализовать(текст: str) -> str:
    """Убирает служебные символы, приводит пробелы и кавычки к единому виду."""
    текст = unicodedata.normalize("NFKC", текст or "")
    текст = _НЕВИДИМЫЕ.sub("", текст)
    текст = _УПРАВЛЯЮЩИЕ.sub(" ", текст)
    текст = текст.replace(" ", " ")
    текст = re.sub(r"[ \t]+", " ", текст)
    текст = re.sub(r"\s*\n\s*", "\n", текст)
    return текст.strip()


def _ключ_дубликата(о: Отзыв) -> tuple:
    полный = " ".join(getattr(о, п).lower() for п in ПОЛЯ)
    return (о.product, о.is_own, re.sub(r"\W+", " ", полный).strip())


def _шинглы(текст: str, n: int = 3) -> set[tuple]:
    слова = re.findall(r"\w+", текст.lower())
    if len(слова) < n:
        return {tuple(слова)} if слова else set()
    return {tuple(слова[i:i + n]) for i in range(len(слова) - n + 1)}


def найти_подозрительные(отзывы: list[Отзыв], *, порог: float = 0.8,
                         мин_слов: int = 6) -> set[str]:
    """id отзывов, у которых есть почти точная копия другого отзыва.

    Короткие тексты («Супер!») пропускаем: у них совпадения случайны.
    Сложность O(n²): для выборок до нескольких тысяч отзывов достаточно.
    """
    кандидаты = []
    for о in отзывы:
        полный = " ".join(getattr(о, п) for п in ПОЛЯ)
        if len(re.findall(r"\w+", полный)) >= мин_слов:
            кандидаты.append((о.id, _шинглы(полный)))
    найдены: set[str] = set()
    for i in range(len(кандидаты)):
        id_a, a = кандидаты[i]
        for j in range(i + 1, len(кандидаты)):
            id_b, b = кандидаты[j]
            if not a or not b:
                continue
            if len(a & b) / len(a | b) >= порог:
                найдены.update((id_a, id_b))
    return найдены


def _фрагменты(источник: str, текст: str) -> list[str]:
    """Предложения поля. В «Достоинствах» и «Недостатках» пункты часто перечисляют
    через «;» («цвет темнее; молния заедает»): это разные мысли, режем и по «;»."""
    предложения = [ф.text.strip() for ф in sentenize(текст)]
    if источник in ("pros", "cons"):
        return [часть.strip() for п in предложения for часть in п.split(";")]
    return предложения


def разбить_на_предложения(о: Отзыв) -> list[Предложение]:
    результат = []
    номер = 0
    for источник in ПОЛЯ:
        for текст in _фрагменты(источник, getattr(о, источник)):
            if not re.search(r"\w", текст):
                continue          # знаки препинания без слов не считаем предложением
            результат.append(Предложение(о.id, номер, текст, источник, о.product, о.is_own))
            номер += 1
    return результат


def предобработать(отзывы: list[Отзыв], *, порог_сходства: float = 0.8) -> Результат:
    """Полная предобработка списка отзывов. Исходные объекты не меняются."""
    всего = len(отзывы)
    очищенные: list[Отзыв] = []
    for о in отзывы:
        копия = Отзыв(**{**о.__dict__, **{п: нормализовать(getattr(о, п)) for п in ПОЛЯ}})
        очищенные.append(копия)

    непустые = [о for о in очищенные if any(getattr(о, п) for п in ПОЛЯ)]
    пустых = всего - len(непустые)

    увиденные: set[tuple] = set()
    уникальные: list[Отзыв] = []
    for о in непустые:
        ключ = _ключ_дубликата(о)
        if ключ in увиденные:
            continue
        увиденные.add(ключ)
        уникальные.append(о)
    дубликатов = len(непустые) - len(уникальные)

    подозрительные = найти_подозрительные(уникальные, порог=порог_сходства)
    предложения = [п for о in уникальные for п in разбить_на_предложения(о)]

    return Результат(
        reviews=уникальные, sentences=предложения, suspicious_ids=подозрительные,
        stats={"loaded": всего, "empty_removed": пустых, "duplicates_removed": дубликатов,
               "kept": len(уникальные), "suspicious": len(подозрительные),
               "sentences": len(предложения)})
