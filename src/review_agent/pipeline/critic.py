"""Шаг 9: цикл «генератор — критик» (FR-15, docs/architecture.md).

Критик — отдельный запрос к модели. Он ставит рекомендации оценки 1–5 по четырём
критериям: конкретность, опора на отзывы, выполнимость, отсутствие повтора.
Решение «прошла / не прошла» принимает код: все оценки не ниже порога (4).
Не прошедшая рекомендация возвращается генератору с замечаниями, не более двух
доработок. Если и после них проверка не пройдена или критик не ответил,
рекомендация в отчёт не попадает, но остаётся в результате для журнала.

Режим без критика (с_критиком=False) нужен для оценки вклада критика (раздел 6 ТЗ).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from review_agent.llm.base import Message
from review_agent.llm.gateway import LLMGateway, LLMGatewayError
from review_agent.pipeline.issues import Проблема
from review_agent.pipeline.recommend import Контекст, Рекомендация, сгенерировать
from review_agent.prompts import загрузить_промпт

ВЕРСИЯ_ПРОМПТА = "critic_v1"
ВЕРСИЯ_ПРОМПТА_СИЛЬНЫХ = "critic_strengths_v1"
КРИТЕРИИ = {"specificity": "конкретность", "grounding": "опора на отзывы",
            "feasibility": "выполнимость", "novelty": "отсутствие повтора"}
_ОЦЕНКА = {"type": "integer", "minimum": 1, "maximum": 5}
СХЕМА = {
    "type": "object",
    "properties": {
        "scores": {"type": "object", "properties": {к: _ОЦЕНКА for к in КРИТЕРИИ},
                   "required": list(КРИТЕРИИ)},
        "comment": {"type": "string"},
    },
    "required": ["scores"],
}


@dataclass
class Вердикт:
    iteration: int                 # 0 — первая версия, 1–2 — доработки
    scores: dict[str, int]
    comment: str
    passed: bool


@dataclass
class Проверенная:
    recommendation: Рекомендация   # последняя версия
    accepted: bool
    verdicts: list[Вердикт] = field(default_factory=list)
    reason: str = ""               # почему отклонена


@dataclass
class РезультатРекомендаций:
    accepted: list[Проверенная] = field(default_factory=list)
    rejected: list[Проверенная] = field(default_factory=list)
    problems_without_recommendation: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def оценить(шлюз: LLMGateway, контекст: Контекст, рек: Рекомендация, принятые: list[str],
            *, итерация: int = 0, порог: int = 4) -> Вердикт:
    """Одна проверка. Бросает LLMGatewayError, если критик не ответил по схеме."""
    версия = ВЕРСИЯ_ПРОМПТА_СИЛЬНЫХ if контекст.kind == "strength" else ВЕРСИЯ_ПРОМПТА
    промпт = загрузить_промпт(версия).format(
        **контекст.поля(), действие=рек.action, тип=рек.type, обоснование=рек.rationale,
        эффект=рек.expected_effect or "не указан", сложность=рек.effort,
        принятые="\n".join(f"- {а}" for а in принятые) or "пока нет", порог=порог)
    r = шлюз.complete("critic", [Message("user", промпт)], schema=СХЕМА,
                      temperature=0.0, max_tokens=500)
    оценки = {к: int(r.data["scores"][к]) for к in КРИТЕРИИ}
    return Вердикт(итерация, оценки, str(r.data.get("comment", "")).strip(),
                   all(в >= порог for в in оценки.values()))


def текст_доработки(рек: Рекомендация, вердикт: Вердикт) -> str:
    оценки = "; ".join(f"{КРИТЕРИИ[к]}: {в} из 5" for к, в in вердикт.scores.items())
    return ("Предыдущая версия рекомендации отклонена экспертом.\n"
            f"Действие: {рек.action}\nОбоснование: {рек.rationale}\n"
            f"Оценки: {оценки}\nЗамечания эксперта: {вердикт.comment or 'не указаны'}\n"
            "Предложи исправленную рекомендацию с учётом замечаний.")


def проверить_и_доработать(шлюз: LLMGateway, контекст: Контекст, рек: Рекомендация,
                           принятые: list[str], *, макс_доработок: int = 2,
                           порог: int = 4) -> Проверенная:
    итог = Проверенная(рек, accepted=False)
    for итерация in range(макс_доработок + 1):
        try:
            вердикт = оценить(шлюз, контекст, итог.recommendation, принятые,
                              итерация=итерация, порог=порог)
        except LLMGatewayError as ошибка:
            итог.reason = f"критик не ответил: {ошибка}"
            return итог
        итог.verdicts.append(вердикт)
        if вердикт.passed:
            итог.accepted = True
            return итог
        if итерация == макс_доработок:
            итог.reason = "не прошла проверку после доработок"
            return итог
        try:
            новые = сгенерировать(шлюз, контекст, количество=1,
                                  доработка=текст_доработки(итог.recommendation, вердикт))
        except LLMGatewayError as ошибка:
            итог.reason = f"генератор не ответил при доработке: {ошибка}"
            return итог
        if not новые:
            итог.reason = "доработанная версия без настоящих цитат"
            return итог
        итог.recommendation = новые[0]
    return итог


def рекомендации_по_проблемам(шлюз: LLMGateway, проблемы: list[Проблема], *, категория: str,
                              всего_отзывов: int, доли_негатива: dict[str, float] | None = None,
                              конкуренты: dict[str, str] | None = None, с_критиком: bool = True,
                              количество: int = 2, макс_доработок: int = 2,
                              порог: int = 4) -> РезультатРекомендаций:
    """Шаги 8–9 для всех проблем. доли_негатива и конкуренты — по названию атрибута."""
    доли_негатива = доли_негатива or {}
    конкуренты = конкуренты or {}
    итог = РезультатРекомендаций()
    с = {"problems": len(проблемы), "drafts": 0, "accepted": 0, "rejected": 0,
         "generator_failed": 0, "revisions": 0, "dropped_quotes": 0, "critic": с_критиком}
    итог.stats = с
    принятые: list[str] = []          # общий список: критик проверяет повторы между проблемами
    for п in проблемы:
        контекст = Контекст(п, категория, всего_отзывов, доли_негатива.get(п.attribute),
                            конкуренты.get(п.attribute, "нет данных"))
        try:
            черновики = сгенерировать(шлюз, контекст, количество=количество)
        except LLMGatewayError:
            с["generator_failed"] += 1
            итог.problems_without_recommendation.append(п.title)
            continue
        с["drafts"] += len(черновики)
        с["dropped_quotes"] += sum(р.dropped_quotes for р in черновики)
        принято_по_проблеме = 0
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
                принято_по_проблеме += 1
            else:
                итог.rejected.append(проверенная)
        if not принято_по_проблеме:
            # Проблема остаётся в отчёте, просто без рекомендации
            итог.problems_without_recommendation.append(п.title)
    с["accepted"], с["rejected"] = len(итог.accepted), len(итог.rejected)
    return итог
