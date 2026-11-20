"""Шаг 6: выделение проблем (docs/architecture.md).

1. Берутся негативные упоминания своего товара.
2. Внутри каждого атрибута цитаты кластеризуются по эмбеддингам.
3. Для каждого кластера LLM формулирует проблему одной фразой по цитатам,
   ближайшим к центру кластера.
4. Кластеры меньше минимального размера (по умолчанию 3, как у Ozon) в отдельные
   проблемы не выделяются.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from review_agent.llm.base import Message
from review_agent.llm.embeddings import Embedder
from review_agent.llm.gateway import LLMGateway, LLMGatewayError
from review_agent.pipeline.cluster import ближайшие_к_центру, кластеризовать
from review_agent.pipeline.extract import Упоминание
from review_agent.prompts import загрузить_промпт

ВЕРСИЯ_ПРОМПТА = "issue_v1"
СХЕМА = {"type": "object",
         "properties": {"title": {"type": "string", "minLength": 3, "maxLength": 150}},
         "required": ["title"]}


@dataclass
class Проблема:
    attribute: str
    title: str
    mentions: int                 # негативных упоминаний в кластере
    reviews: int                  # разных отзывов
    coverage: float               # reviews / все отзывы своего товара
    quotes: list[str]             # характерные цитаты (ближайшие к центру)
    review_ids: list[str]         # для проверки опоры на отзывы (шаг 11)
    title_ok: bool = True         # False: название не получено от модели


@dataclass
class РезультатПроблем:
    problems: list[Проблема] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def _уникальные(цитаты: list[str]) -> list[str]:
    """Цитаты без повторов (регистр и знаки препинания не учитываются), порядок сохраняется."""
    увиденные: set[str] = set()
    результат = []
    for ц in цитаты:
        ключ = re.sub(r"\W+", " ", ц.lower()).strip()
        if ключ not in увиденные:
            увиденные.add(ключ)
            результат.append(ц)
    return результат


def выделить_проблемы(шлюз: LLMGateway, эмбеддер: Embedder, упоминания: list[Упоминание],
                      *, категория: str, всего_отзывов: int, порог: float = 0.35,
                      мин_размер: int = 3, цитат_для_названия: int = 8,
                      цитат_в_отчёт: int = 5) -> РезультатПроблем:
    """Находит проблемы своего товара. всего_отзывов нужен для охвата."""
    негативные = [м for м in упоминания if м.is_own and м.sentiment == "negative"]
    статистика = {"negative_mentions": len(негативные), "clusters": 0, "problems": 0,
                  "small_clusters": 0, "mentions_unassigned": 0, "titles_failed": 0}
    итог = РезультатПроблем(stats=статистика)
    if not негативные:
        return итог

    # эмбеддинги считаются один раз для каждого уникального текста
    тексты = sorted({м.text for м in негативные})
    векторы_по_тексту = dict(zip(тексты, эмбеддер.embed(тексты)))

    по_атрибутам: dict[str, list[Упоминание]] = defaultdict(list)
    for м in негативные:
        по_атрибутам[м.attribute].append(м)

    шаблон = загрузить_промпт(ВЕРСИЯ_ПРОМПТА)
    for атрибут, список in по_атрибутам.items():
        векторы = [векторы_по_тексту[м.text] for м in список]
        for кластер in кластеризовать(векторы, порог):
            статистика["clusters"] += 1
            if len(кластер) < мин_размер:
                статистика["small_clusters"] += 1
                статистика["mentions_unassigned"] += len(кластер)
                continue
            ближайшие = ближайшие_к_центру(векторы, кластер, len(кластер))
            цитаты = _уникальные([список[i].text for i in ближайшие])
            название, ок = _назвать(шлюз, шаблон, категория, атрибут, цитаты[:цитат_для_названия])
            if not ок:
                статистика["titles_failed"] += 1
            id_отзывов = sorted({список[i].review_id for i in кластер})
            итог.problems.append(Проблема(
                attribute=атрибут, title=название, mentions=len(кластер), reviews=len(id_отзывов),
                coverage=len(id_отзывов) / всего_отзывов if всего_отзывов else 0.0,
                quotes=цитаты[:цитат_в_отчёт], review_ids=id_отзывов, title_ok=ок))
    итог.problems.sort(key=lambda п: (-п.reviews, -п.mentions, п.attribute))
    статистика["problems"] = len(итог.problems)
    return итог


def _назвать(шлюз: LLMGateway, шаблон: str, категория: str, атрибут: str,
             цитаты: list[str]) -> tuple[str, bool]:
    список = "\n".join(f"- {ц}" for ц in цитаты)
    промпт = шаблон.format(категория=категория, атрибут=атрибут, цитаты=список)
    try:
        r = шлюз.complete("issue_title", [Message("user", промпт)], schema=СХЕМА,
                          temperature=0.0, max_tokens=200)
        return r.data["title"].strip(), True
    except LLMGatewayError:
        return f"{атрибут}: название не получено", False
