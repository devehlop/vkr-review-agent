"""Шаг 8: генерация рекомендаций по проблемам (FR-13, FR-14, docs/architecture.md).

Генератор получает проблему из шага 6, её статистику, характерные цитаты и
данные конкурентов (пока «нет данных»: шаг 7 не сделан) и предлагает 1–2 действия:
действие, тип, обоснование, ожидаемый эффект, сложность 1–3 и цитаты-опору.

Цитаты проверяются кодом: фрагмент должен дословно (без учёта регистра и
пробелов) входить в цитату с указанным номером. Выдуманные цитаты отбрасываются,
рекомендация без единой настоящей цитаты в результат не попадает (FR-17).

Рекомендации по сильным сторонам (вторая часть FR-13) будут отдельной функцией.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from review_agent.llm.base import Message
from review_agent.llm.gateway import LLMGateway
from review_agent.pipeline.issues import Проблема
from review_agent.prompts import загрузить_промпт

ВЕРСИЯ_ПРОМПТА = "recommend_v1"
ТИПЫ = ("товар", "упаковка", "карточка товара", "размерная сетка", "логистика")
НЕТ_КОНКУРЕНТОВ = "нет данных"

СХЕМА = {
    "type": "object",
    "properties": {"recommendations": {"type": "array", "minItems": 1, "maxItems": 3, "items": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "minLength": 5},
            "type": {"enum": list(ТИПЫ)},
            "rationale": {"type": "string", "minLength": 5},
            "expected_effect": {"type": "string"},
            "effort": {"type": "integer", "minimum": 1, "maximum": 3},
            "quotes": {"type": "array", "minItems": 1, "maxItems": 3, "items": {
                "type": "object",
                "properties": {"n": {"type": "integer", "minimum": 1},
                               "text": {"type": "string", "minLength": 2}},
                "required": ["n", "text"]}},
        },
        "required": ["action", "type", "rationale", "effort", "quotes"]}}},
    "required": ["recommendations"],
}


@dataclass
class Рекомендация:
    problem: str                  # название проблемы из шага 6
    attribute: str
    action: str
    type: str
    rationale: str
    expected_effect: str
    effort: int                   # 1 — карточка товара … 3 — сам товар
    quotes: list[str]             # проверенные фрагменты цитат
    review_ids: list[str] = field(default_factory=list)   # отзывы проблемы (для шага 11)
    dropped_quotes: int = 0       # сколько цитат модели не нашлось в тексте


@dataclass
class Контекст:
    """Всё, что генератор и критик знают о проблеме."""
    problem: Проблема
    category: str
    total_reviews: int
    negative_share: float | None = None      # доля негатива по атрибуту (шаг 5)
    competitors: str = НЕТ_КОНКУРЕНТОВ        # резюме сравнения (шаг 7)

    def цитаты(self) -> str:
        return "\n".join(f"{i}. {ц}" for i, ц in enumerate(self.problem.quotes, 1))

    def поля(self) -> dict:
        п = self.problem
        доля = "нет данных" if self.negative_share is None else f"{self.negative_share:.0%}"
        return {"категория": self.category, "проблема": п.title, "атрибут": п.attribute,
                "отзывов": п.reviews, "всего": self.total_reviews, "охват": f"{п.coverage:.0%}",
                "доля_негатива": доля, "конкуренты": self.competitors, "цитаты": self.цитаты()}


def _ключ(текст: str) -> str:
    return re.sub(r"\s+", " ", текст.lower().replace("ё", "е")).strip(" .,!?;:«»\"'")


def проверить_цитаты(цитаты_модели: list[dict], цитаты_проблемы: list[str]) -> tuple[list[str], int]:
    """Оставляет фрагменты, которые дословно есть в цитате с указанным номером.

    Возвращает (проверенные фрагменты, число отброшенных).
    """
    верные: list[str] = []
    отброшено = 0
    for ц in цитаты_модели:
        n, фрагмент = ц.get("n", 0), str(ц.get("text", ""))
        if 1 <= n <= len(цитаты_проблемы) and _ключ(фрагмент) and \
                _ключ(фрагмент) in _ключ(цитаты_проблемы[n - 1]):
            if фрагмент not in верные:
                верные.append(фрагмент)
        else:
            отброшено += 1
    return верные, отброшено


def сгенерировать(шлюз: LLMGateway, контекст: Контекст, *, количество: int = 2,
                  доработка: str = "") -> list[Рекомендация]:
    """Один запрос к генератору. Бросает LLMGatewayError, если модель не ответила по схеме.

    доработка — замечания критика к предыдущей версии; тогда просится одна рекомендация.
    """
    п = контекст.problem
    сколько = "одно действие" if количество == 1 else f"{количество} действия"
    промпт = загрузить_промпт(ВЕРСИЯ_ПРОМПТА).format(
        **контекст.поля(), типы=", ".join(ТИПЫ), количество=сколько,
        доработка=f"\n{доработка}\n" if доработка else "")
    r = шлюз.complete("recommend", [Message("user", промпт)], schema=СХЕМА,
                      temperature=0.2, max_tokens=1500)
    итог: list[Рекомендация] = []
    for д in r.data["recommendations"][:количество]:
        цитаты, отброшено = проверить_цитаты(д["quotes"], п.quotes)
        if not цитаты:
            continue          # ни одной настоящей цитаты: рекомендация без опоры
        итог.append(Рекомендация(
            problem=п.title, attribute=п.attribute, action=д["action"].strip(), type=д["type"],
            rationale=д["rationale"].strip(), expected_effect=д.get("expected_effect", "").strip(),
            effort=д["effort"], quotes=цитаты, review_ids=list(п.review_ids),
            dropped_quotes=отброшено))
    return итог

