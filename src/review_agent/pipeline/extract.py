"""Шаг 4: извлечение аспектов по предложениям (docs/architecture.md).

Предложения отправляются модели пакетами. Для каждого предложения модель возвращает
список пар «аспект из справочника + тональность». Ответ проверяется по схеме и по
справочнику; пакет, который не удалось обработать, делится пополам и половины обрабатываются
отдельно; предложение, не прошедшее даже поодиночке, пропускается и учитывается в
статистике (NFR-6), остальные продолжают обрабатываться.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from review_agent.llm.base import Message
from review_agent.llm.gateway import LLMGateway, LLMGatewayError
from review_agent.pipeline.taxonomy import Справочник
from review_agent.preprocess.cleaner import Предложение
from review_agent.prompts import загрузить_промпт

ВЕРСИЯ_ПРОМПТА = "extract_v1"
ТОНАЛЬНОСТИ = ("positive", "negative", "neutral")
_МЕТКА_ИСТОЧНИКА = {"pros": "плюсы", "cons": "минусы", "text": "отзыв"}

СХЕМА = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "i": {"type": "integer", "minimum": 1},
                    "mentions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "attribute": {"type": "string"},
                                "sentiment": {"enum": list(ТОНАЛЬНОСТИ)},
                            },
                            "required": ["attribute", "sentiment"],
                        },
                    },
                },
                "required": ["i", "mentions"],
            },
        }
    },
    "required": ["items"],
}


@dataclass
class Упоминание:
    review_id: str
    sentence_idx: int
    attribute: str
    sentiment: str
    source: str          # откуда предложение: text / pros / cons
    text: str            # само предложение (для цитат в отчёте, шаг 11)
    product: str
    is_own: bool


@dataclass
class РезультатИзвлечения:
    mentions: list[Упоминание] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def _описание_справочника(с: Справочник) -> str:
    return "\n".join(f"- {а.name}: {', '.join(а.features)}" for а in с.attributes)


def _пакеты(предложения: list[Предложение], размер: int):
    for начало in range(0, len(предложения), размер):
        yield предложения[начало:начало + размер]


def _разобрать_пакет(пакет: list[Предложение], данные: dict, допустимые: dict[str, str],
                     статистика: dict) -> list[Упоминание]:
    """Превращает ответ модели в упоминания, отбрасывая то, чего нет в справочнике."""
    результат = []
    for элемент in данные["items"]:
        номер = элемент["i"]
        if not 1 <= номер <= len(пакет):
            статистика["bad_index"] += 1
            continue
        п = пакет[номер - 1]
        уже: set[tuple[str, str]] = set()
        for м in элемент["mentions"]:
            атрибут = допустимые.get(м["attribute"].strip().lower())
            if атрибут is None:
                статистика["unknown_attribute"] += 1
                continue
            if (атрибут, м["sentiment"]) in уже:
                continue
            уже.add((атрибут, м["sentiment"]))
            результат.append(Упоминание(п.review_id, п.idx, атрибут, м["sentiment"],
                                        п.source, п.text, п.product, п.is_own))
    return результат


def извлечь_аспекты(шлюз: LLMGateway, предложения: list[Предложение], справочник: Справочник,
                    *, размер_пакета: int = 15) -> РезультатИзвлечения:
    """Извлекает упоминания аспектов для всех предложений."""
    шаблон = загрузить_промпт(ВЕРСИЯ_ПРОМПТА)
    описание = _описание_справочника(справочник)
    допустимые = {а.name.strip().lower(): а.name for а in справочник.attributes}
    статистика = {"sentences": len(предложения), "batches": 0, "batches_failed": 0, "batches_split": 0,
                  "sentences_skipped": 0, "unknown_attribute": 0, "bad_index": 0}
    итог = РезультатИзвлечения(stats=статистика)

    def обработать(пакет: list[Предложение], уровень: int = 0) -> None:
        """Отправляет пакет; при сбое делит пополам и пробует половины отдельно."""
        статистика["batches"] += 1
        список = "\n".join(f"{i}. [{_МЕТКА_ИСТОЧНИКА[п.source]}] {п.text}"
                           for i, п in enumerate(пакет, 1))
        промпт = шаблон.format(категория=справочник.category, справочник=описание,
                               предложения=список)
        try:
            ответ = шлюз.complete("extract", [Message("user", промпт)], schema=СХЕМА,
                                  temperature=0.0, max_tokens=3000)
        except LLMGatewayError:
            if len(пакет) > 1:
                статистика["batches_split"] += 1
                середина = len(пакет) // 2
                обработать(пакет[:середина], уровень + 1)
                обработать(пакет[середина:], уровень + 1)
            else:
                статистика["batches_failed"] += 1
                статистика["sentences_skipped"] += 1
            return
        итог.mentions.extend(_разобрать_пакет(пакет, ответ.data, допустимые, статистика))

    for пакет in _пакеты(предложения, размер_пакета):
        обработать(пакет)
    статистика["mentions"] = len(итог.mentions)
    return итог
