"""Шаг 12: сборка отчёта для продавца из результатов шагов 5–11 (FR-18).

Текст отчёта собирает код из уже проверенных данных, модель здесь не вызывается.
Поэтому в отчёт не может попасть утверждение, не прошедшее проверки шагов 9 и 11.

Источники (те же файлы, что пишут скрипты шагов):
- отзывы после предобработки — число отзывов и подозрительных;
- aggregate.csv (шаг 5) — характеристики, сильные стороны, доля негатива;
- issues.json (шаг 6) — проблемы и их цитаты;
- recommendations.json (шаги 8–9) — отклонённые критиком, проблемы без рекомендаций,
  рекомендации по сильным сторонам (если были);
- prioritized.json (шаги 10–11) — рекомендации по приоритету и отклонённые проверкой опоры.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

МИН_ОТЗЫВОВ = 30            # меньше — выводы ненадёжны (ТЗ, ограничения)
МИН_УПОМИНАНИЙ_СИЛЬНОЙ = 3  # тот же порог, что у сильных сторон на шаге 8
МИН_ДОЛЯ_ПОЗИТИВА = 0.6
ЦИТАТ_У_ПРОБЛЕМЫ = 3
СЛОЖНОСТЬ = {1: "низкая", 2: "средняя", 3: "высокая"}


@dataclass
class Характеристика:
    attribute: str
    mentions: int
    coverage: float
    positive: int
    negative: int
    neutral: int
    negative_share: float
    mean_rating: float | None


@dataclass
class ПунктРекомендации:
    action: str
    problem: str
    attribute: str
    type: str
    effort: int
    priority: float
    coverage: float
    negative_share: float | None      # None: доли негатива не было, в формуле взята 1,0
    rationale: str
    expected_effect: str
    quotes: list[str]


@dataclass
class ПунктПроблемы:
    title: str
    attribute: str
    reviews: int
    coverage: float
    quotes: list[str]
    has_recommendation: bool


@dataclass
class СильнаяСторона:
    attribute: str
    coverage: float
    positive_share: float
    card_tip: str = ""                # что подчеркнуть в карточке (если шаг 8 его дал)
    quotes: list[str] = field(default_factory=list)


@dataclass
class Отчёт:
    product: str
    created: date
    reviews: int
    suspicious: int
    recommendations: list[ПунктРекомендации]
    problems: list[ПунктПроблемы]
    strengths: list[СильнаяСторона]
    characteristics: list[Характеристика]
    rejected: list[tuple[str, str, str]]        # (проблема, действие, причина) — для журнала
    critic: bool | None                          # был ли включён критик
    warnings: list[str] = field(default_factory=list)


def без_повторов(цитаты: list[str]) -> list[str]:
    """Цитаты без повторов: регистр, «ё», пробелы и знаки препинания не учитываются.

    В issues.json одна и та же фраза из разных отзывов может стоять несколько раз
    (например, «Коробка помята» в четырёх отзывах); в отчёте её достаточно показать один раз.
    """
    увиденные: set[str] = set()
    итог = []
    for ц in цитаты:
        ключ = re.sub(r"\W+", " ", ц.lower().replace("ё", "е")).strip()
        if ключ and ключ not in увиденные:
            увиденные.add(ключ)
            итог.append(ц)
    return итог


def прочитать_агрегат(путь: str | Path, товар: str) -> list[Характеристика]:
    """Строки aggregate.csv своего товара."""
    итог = []
    with open(путь, encoding="utf-8", newline="") as f:
        for с in csv.DictReader(f):
            if с["product"] != товар or с["is_own"] not in ("1", "True", "true"):
                continue
            итог.append(Характеристика(
                attribute=с["attribute"], mentions=int(с["mentions"]), coverage=float(с["coverage"]),
                positive=int(с["positive"]), negative=int(с["negative"]), neutral=int(с["neutral"]),
                negative_share=float(с["negative_share"]),
                mean_rating=float(с["mean_rating"]) if с.get("mean_rating") else None))
    return итог


def _json(путь: str | Path | None, по_умолчанию):
    if not путь:
        return по_умолчанию
    return json.loads(Path(путь).read_text(encoding="utf-8"))


def собрать(*, товар: str, отзывов: int, подозрительных: int = 0,
            характеристики: list[Характеристика], проблемы: list[dict],
            приоритеты: dict, рекомендации: dict | None = None,
            дата: date | None = None) -> Отчёт:
    """Собирает отчёт из уже загруженных данных (словари — как в JSON-файлах шагов)."""
    рекомендации = рекомендации or {}
    пункты = []
    for п in приоритеты.get("prioritized", []):
        р = п["recommendation"]
        пункты.append(ПунктРекомендации(
            action=р["action"], problem=р["problem"], attribute=р["attribute"], type=р["type"],
            effort=р["effort"], priority=п["priority"], coverage=п["coverage"],
            negative_share=п["negative_share"] if п.get("negative_share_known", True) else None,
            rationale=р.get("rationale", ""), expected_effect=р.get("expected_effect", ""),
            quotes=без_повторов(р.get("quotes", []))))

    с_рекомендацией = {п.problem for п in пункты}
    пункты_проблем = [ПунктПроблемы(
        title=п["title"], attribute=п["attribute"], reviews=п["reviews"], coverage=п["coverage"],
        quotes=без_повторов(п["quotes"])[:ЦИТАТ_У_ПРОБЛЕМЫ], has_recommendation=п["title"] in с_рекомендацией)
        for п in sorted(проблемы, key=lambda п: (-п["coverage"], п["title"]))]

    # Сильные стороны: характеристики, которые заметно чаще хвалят, без найденных проблем.
    # Отбор такой же, как на шаге 8 (pipeline/strengths.py), чтобы отчёт не противоречил советам.
    атрибуты_проблем = {п["attribute"] for п in проблемы}
    советы = {п["recommendation"]["problem"]: п["recommendation"]
              for п in рекомендации.get("strengths", {}).get("accepted", [])}
    сильные = []
    for х in характеристики:
        if х.attribute in атрибуты_проблем or х.mentions < МИН_УПОМИНАНИЙ_СИЛЬНОЙ:
            continue
        доля = х.positive / х.mentions
        if доля >= МИН_ДОЛЯ_ПОЗИТИВА:
            совет = советы.get(х.attribute, {})
            сильные.append(СильнаяСторона(х.attribute, х.coverage, доля, совет.get("action", ""),
                                          list(совет.get("quotes", []))))
    сильные.sort(key=lambda с: (-с.coverage * с.positive_share, с.attribute))

    отклонённые = [(п["recommendation"]["problem"], п["recommendation"]["action"],
                    f"критик: {п.get('reason') or 'не прошла проверку'}")
                   for п in рекомендации.get("rejected", [])]
    отклонённые += [(о["problem"], о["action"], f"нет опоры на отзывы: {о['reason']}")
                    for о in приоритеты.get("ungrounded", [])]

    предупреждения = []
    if отзывов < МИН_ОТЗЫВОВ:
        предупреждения.append(f"Отзывов всего {отзывов}: выводы ненадёжны, "
                              f"нужно не меньше {МИН_ОТЗЫВОВ}.")
    if подозрительных:
        предупреждения.append(f"Подозрительных отзывов (почти одинаковый текст): {подозрительных}. "
                              "Они учтены в анализе.")
    без_доли = приоритеты.get("stats", {}).get("negative_share_missing") or []
    if без_доли:
        предупреждения.append("Для характеристик " + ", ".join(без_доли) + " не было доли негатива: "
                              "их приоритет посчитан только по охвату и сложности.")
    if рекомендации.get("stats", {}).get("critic") is False:
        предупреждения.append("Рекомендации не проверялись критиком.")

    return Отчёт(product=товар, created=дата or date.today(), reviews=отзывов,
                 suspicious=подозрительных, recommendations=пункты, problems=пункты_проблем,
                 strengths=сильные, characteristics=характеристики, rejected=отклонённые,
                 critic=рекомендации.get("stats", {}).get("critic"), warnings=предупреждения)


def собрать_из_файлов(*, товар: str, отзывов: int, подозрительных: int = 0,
                      агрегат: str | Path, проблемы: str | Path, приоритеты: str | Path,
                      рекомендации: str | Path | None = None, дата: date | None = None) -> Отчёт:
    return собрать(товар=товар, отзывов=отзывов, подозрительных=подозрительных,
                   характеристики=прочитать_агрегат(агрегат, товар), проблемы=_json(проблемы, []),
                   приоритеты=_json(приоритеты, {}), рекомендации=_json(рекомендации, {}), дата=дата)


def процент(доля: float) -> str:
    return f"{доля * 100:.0f} %"
