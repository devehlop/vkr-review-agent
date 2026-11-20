"""Шаг 3: справочник аспектов (docs/architecture.md).

LLM получает случайную выборку отзывов и предлагает справочник «атрибут → признаки».
Справочник сохраняется в JSON: продавец может его поправить вручную. Один справочник
используется и для своего товара, и для конкурентов, иначе сравнение невозможно.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from review_agent.ingest.loader import Отзыв
from review_agent.llm.base import Message
from review_agent.llm.gateway import LLMGateway
from review_agent.prompts import загрузить_промпт

ВЕРСИЯ_ПРОМПТА = "taxonomy_v2"
МАКС_ДЛИНА_ОТЗЫВА = 400   # символов; длинные отзывы обрезаем, чтобы не раздувать запрос

СХЕМА = {
    "type": "object",
    "properties": {
        "attributes": {
            "type": "array", "minItems": 3, "maxItems": 15,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "features": {"type": "array", "minItems": 1, "items": {"type": "string"}},
                },
                "required": ["name", "features"],
            },
        }
    },
    "required": ["attributes"],
}


@dataclass
class Атрибут:
    name: str
    features: list[str] = field(default_factory=list)


@dataclass
class Справочник:
    category: str
    attributes: list[Атрибут]
    prompt_version: str = ВЕРСИЯ_ПРОМПТА

    def сохранить(self, путь: str | Path) -> None:
        данные = {"category": self.category, "prompt_version": self.prompt_version,
                  "attributes": [{"name": a.name, "features": a.features} for a in self.attributes]}
        Path(путь).write_text(json.dumps(данные, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def загрузить(cls, путь: str | Path) -> "Справочник":
        д = json.loads(Path(путь).read_text(encoding="utf-8"))
        return cls(д["category"], [Атрибут(а["name"], а["features"]) for а in д["attributes"]],
                   д.get("prompt_version", ""))


def _текст_отзыва(о: Отзыв) -> str:
    части = []
    if о.pros:
        части.append(f"Достоинства: {о.pros}")
    if о.cons:
        части.append(f"Недостатки: {о.cons}")
    if о.text:
        части.append(о.text)
    return " ".join(части)[:МАКС_ДЛИНА_ОТЗЫВА]


def выбрать_выборку(отзывы: list[Отзыв], размер: int = 150, seed: int = 42) -> list[Отзыв]:
    """Случайная воспроизводимая выборка (тот же seed даёт ту же выборку)."""
    if len(отзывы) <= размер:
        return list(отзывы)
    return random.Random(seed).sample(отзывы, размер)


def _убрать_повторы(атрибуты: list[Атрибут]) -> list[Атрибут]:
    """Объединяет атрибуты с одинаковым названием и чистит повторы признаков."""
    по_имени: dict[str, Атрибут] = {}
    for а in атрибуты:
        ключ = а.name.strip().lower()
        цель = по_имени.setdefault(ключ, Атрибут(а.name.strip(), []))
        for п in а.features:
            п = п.strip()
            if п and п.lower() not in {x.lower() for x in цель.features}:
                цель.features.append(п)
    return list(по_имени.values())


def построить_справочник(шлюз: LLMGateway, отзывы: list[Отзыв], *, категория: str,
                         размер_выборки: int = 150, seed: int = 42) -> Справочник:
    """Просит модель составить справочник по выборке отзывов."""
    выборка = [о for о in выбрать_выборку(отзывы, размер_выборки, seed) if _текст_отзыва(о)]
    if not выборка:
        raise ValueError("Нет отзывов с текстом для построения справочника")
    список = "\n".join(f"{i}. {_текст_отзыва(о)}" for i, о in enumerate(выборка, 1))
    промпт = загрузить_промпт(ВЕРСИЯ_ПРОМПТА).format(категория=категория, отзывы=список)
    результат = шлюз.complete("taxonomy", [Message("user", промпт)],
                              schema=СХЕМА, temperature=0.0, max_tokens=2000)
    атрибуты = _убрать_повторы(
        [Атрибут(а["name"], а["features"]) for а in результат.data["attributes"]])
    return Справочник(категория, атрибуты)
