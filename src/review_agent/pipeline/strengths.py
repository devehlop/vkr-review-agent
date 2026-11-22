"""Шаг 8, вторая часть FR-13: рекомендации по сильным сторонам товара.

Сильная сторона — характеристика своего товара, которую покупатели заметно чаще
хвалят, чем ругают. По ней генератор предлагает, что подчеркнуть в карточке товара,
а критик проверяет так же, как рекомендации по проблемам (pipeline/critic.py).

Характеристики, по которым на шаге 6 найдены проблемы, в сильные стороны не попадают:
иначе продавец получит противоречивые советы («подчеркнуть быструю доставку» рядом
с проблемой «задержка доставки»). Это выяснилось на пробном прогоне по data/issues.json.

Сильная сторона передаётся в генератор и критик как Проблема (тема + позитивные цитаты
+ id отзывов для шага 11) с Контекст.kind = "strength".
"""
from __future__ import annotations

import re

from review_agent.llm.gateway import LLMGateway, LLMGatewayError
from review_agent.pipeline.aggregate import СтатАспекта
from review_agent.pipeline.critic import Проверенная, РезультатРекомендаций, проверить_и_доработать
from review_agent.pipeline.extract import Упоминание
from review_agent.pipeline.issues import Проблема
from review_agent.pipeline.recommend import Контекст, сгенерировать


def _уникальные(тексты: list[str]) -> list[str]:
    увиденные: set[str] = set()
    итог = []
    for т in тексты:
        ключ = re.sub(r"\W+", " ", т.lower().replace("ё", "е")).strip()
        if ключ and ключ not in увиденные:
            увиденные.add(ключ)
            итог.append(т)
    return итог


def найти_сильные_стороны(статистика: list[СтатАспекта], упоминания: list[Упоминание], *,
                          исключить: set[str] = frozenset(), мин_доля: float = 0.6,
                          мин_упоминаний: int = 3, лимит: int = 3,
                          цитат: int = 5) -> list[tuple[Проблема, float]]:
    """Сильные стороны своего товара: (тема с позитивными цитатами, доля позитива).

    Отбор: свой товар, не меньше мин_упоминаний, доля позитива не ниже мин_доля,
    тема не из исключить. Порядок: по охвату × доле позитива.
    """
    кандидаты = [с for с in статистика
                 if с.is_own and с.attribute not in исключить and с.mentions >= мин_упоминаний
                 and с.positive / с.mentions >= мин_доля]
    кандидаты.sort(key=lambda с: (-с.coverage * с.positive / с.mentions, с.attribute))
    итог = []
    for с in кандидаты[:лимит]:
        позитивные = [м for м in упоминания
                      if м.is_own and м.attribute == с.attribute and м.sentiment == "positive"]
        тема = Проблема(attribute=с.attribute, title=с.attribute, mentions=с.positive, reviews=с.reviews,
                        coverage=с.coverage, quotes=_уникальные([м.text for м in позитивные])[:цитат],
                        review_ids=sorted({м.review_id for м in позитивные}))
        итог.append((тема, с.positive / с.mentions))
    return итог


def рекомендации_по_сильным_сторонам(шлюз: LLMGateway, стороны: list[tuple[Проблема, float]], *,
                                     категория: str, всего_отзывов: int,
                                     принятые: list[str] | None = None, с_критиком: bool = True,
                                     макс_доработок: int = 2, порог: int = 4) -> РезультатРекомендаций:
    """По одной рекомендации для карточки товара на каждую сильную сторону.

    принятые — действия, уже принятые по проблемам: критик проверяет повторы и с ними.
    """
    принятые = list(принятые or [])
    итог = РезультатРекомендаций()
    с = {"strengths": len(стороны), "drafts": 0, "accepted": 0, "rejected": 0,
         "generator_failed": 0, "revisions": 0, "dropped_quotes": 0, "critic": с_критиком}
    итог.stats = с
    for тема, доля in стороны:
        контекст = Контекст(тема, категория, всего_отзывов, kind="strength", positive_share=доля)
        try:
            черновики = сгенерировать(шлюз, контекст, количество=1)
        except LLMGatewayError:
            с["generator_failed"] += 1
            итог.problems_without_recommendation.append(тема.title)
            continue
        с["drafts"] += len(черновики)
        с["dropped_quotes"] += sum(р.dropped_quotes for р in черновики)
        if not черновики:
            итог.problems_without_recommendation.append(тема.title)
        for рек in черновики:
            if с_критиком:
                проверенная = проверить_и_доработать(шлюз, контекст, рек, принятые,
                                                     макс_доработок=макс_доработок, порог=порог)
                с["revisions"] += max(0, len(проверенная.verdicts) - 1)
            else:
                проверенная = Проверенная(рек, accepted=True)
            if проверенная.accepted:
                итог.accepted.append(проверенная)
                принятые.append(проверенная.recommendation.action)
            else:
                итог.rejected.append(проверенная)
                итог.problems_without_recommendation.append(тема.title)
    с["accepted"], с["rejected"] = len(итог.accepted), len(итог.rejected)
    return итог
