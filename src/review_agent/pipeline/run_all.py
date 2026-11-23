"""Полный прогон конвейера: от отзывов до отчёта (шаги 2–12).

Нужен интерфейсу: скрипты scripts/run_*.py запускают шаги по одному и обмениваются
файлами, а здесь те же функции вызываются подряд в памяти. Результат можно сохранить
в папку в тех же форматах, что пишут скрипты, и потом открыть снова без модели.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from review_agent.ingest.loader import Отзыв
from review_agent.llm.embeddings import Embedder
from review_agent.llm.gateway import LLMGateway
from review_agent.pipeline.aggregate import СтатАспекта, агрегировать, сохранить_csv
from review_agent.pipeline.critic import РезультатРекомендаций, рекомендации_по_проблемам
from review_agent.pipeline.extract import Упоминание, извлечь_аспекты
from review_agent.pipeline.grounding import Отказ, проверить_опору
from review_agent.pipeline.issues import Проблема, выделить_проблемы
from review_agent.pipeline.prioritize import Приоритетная, расставить
from review_agent.pipeline.taxonomy import Справочник, построить_справочник
from review_agent.preprocess.cleaner import предобработать
from review_agent.report.build import Отчёт, Характеристика, собрать
from review_agent.report.render import в_docx, в_markdown


@dataclass
class ИтогАнализа:
    product: str
    taxonomy: Справочник
    reviews: list[Отзыв]
    preprocess: dict
    mentions: list[Упоминание]
    aggregate: list[СтатАспекта]
    problems: list[Проблема]
    recommendations: РезультатРекомендаций
    prioritized: list[Приоритетная]
    ungrounded: list[Отказ]
    report: Отчёт
    stats: dict = field(default_factory=dict)       # по шагам и по вызовам модели
    errors: list[tuple[str, str]] = field(default_factory=list)


def характеристики(статистика: list[СтатАспекта], товар: str) -> list[Характеристика]:
    """Строки агрегата своего товара в виде, который нужен отчёту."""
    return [Характеристика(с.attribute, с.mentions, с.coverage, с.positive, с.negative, с.neutral,
                           с.negative_share, с.mean_rating)
            for с in статистика if с.is_own and с.product == товар]


def доли_негатива(статистика: list[СтатАспекта], товар: str) -> dict[str, float]:
    return {с.attribute: с.negative_share for с in статистика if с.is_own and с.product == товар}


def словари_для_отчёта(итог: РезультатРекомендаций, приоритеты: list[Приоритетная],
                       отказы: list[Отказ], доли: dict[str, float]) -> tuple[dict, dict]:
    """recommendations.json и prioritized.json в том виде, как их пишут скрипты шагов 8–11."""
    рекомендации = {"stats": итог.stats, "accepted": [asdict(п) for п in итог.accepted],
                    "rejected": [asdict(п) for п in итог.rejected],
                    "problems_without_recommendation": итог.problems_without_recommendation}
    без_доли = sorted({п.recommendation.attribute for п in приоритеты if not п.negative_share_known})
    приоритетные = {"stats": {"recommendations": len(итог.accepted), "grounded": len(приоритеты),
                              "ungrounded": len(отказы), "critic": итог.stats.get("critic"),
                              "negative_share_missing": без_доли},
                    "prioritized": [asdict(п) for п in приоритеты],
                    "ungrounded": [asdict(о) for о in отказы]}
    return рекомендации, приоритетные


def запустить_анализ(шлюз: LLMGateway, эмбеддер: Embedder, отзывы: list[Отзыв], *, товар: str,
                     категория: str, справочник: Справочник | None = None, с_критиком: bool = True,
                     порог_кластеризации: float = 0.35, мин_размер_проблемы: int = 3,
                     шаг: Callable[[str], None] | None = None) -> ИтогАнализа:
    """Все шаги подряд. шаг(текст) сообщает интерфейсу, что сейчас выполняется."""
    сообщить = шаг or (lambda текст: None)

    сообщить("Предобработка отзывов")
    предобработка = предобработать(отзывы)
    свои = [о for о in предобработка.reviews if о.is_own and о.product == товар]
    if not свои:
        raise ValueError(f"Нет отзывов на товар «{товар}» после предобработки")

    if справочник is None:
        сообщить("Справочник аспектов (модель)")
        справочник = построить_справочник(шлюз, предобработка.reviews, категория=категория)

    сообщить(f"Извлечение аспектов: {len(предобработка.sentences)} предложений (модель)")
    извлечение = извлечь_аспекты(шлюз, предобработка.sentences, справочник)

    сообщить("Агрегация")
    статистика = агрегировать(предобработка.reviews, извлечение.mentions,
                              [а.name for а in справочник.attributes])
    доли = доли_негатива(статистика, товар)

    сообщить("Выделение проблем (эмбеддинги и модель)")
    проблемы = выделить_проблемы(шлюз, эмбеддер, извлечение.mentions, категория=категория,
                                 всего_отзывов=len(свои), порог=порог_кластеризации,
                                 мин_размер=мин_размер_проблемы)

    сообщить("Рекомендации" + (" и проверка критиком" if с_критиком else "") + " (модель)")
    рекомендации = рекомендации_по_проблемам(шлюз, проблемы.problems, категория=категория,
                                             всего_отзывов=len(свои), доли_негатива=доли,
                                             с_критиком=с_критиком)

    сообщить("Проверка опоры на отзывы и приоритеты")
    прошли, отказы = проверить_опору([п.recommendation for п in рекомендации.accepted],
                                     проблемы.problems, предобработка.reviews)
    приоритеты = расставить(прошли, проблемы.problems, доли_негатива=доли)

    сообщить("Отчёт")
    словарь_рек, словарь_приор = словари_для_отчёта(рекомендации, приоритеты, отказы, доли)
    отчёт = собрать(товар=товар, отзывов=len(свои), подозрительных=len(предобработка.suspicious_ids),
                    характеристики=характеристики(статистика, товар),
                    проблемы=[asdict(п) for п in проблемы.problems],
                    приоритеты=словарь_приор, рекомендации=словарь_рек)

    return ИтогАнализа(
        product=товар, taxonomy=справочник, reviews=предобработка.reviews,
        preprocess=предобработка.stats, mentions=извлечение.mentions, aggregate=статистика,
        problems=проблемы.problems, recommendations=рекомендации, prioritized=приоритеты,
        ungrounded=отказы, report=отчёт,
        stats={"extract": извлечение.stats, "issues": проблемы.stats,
               "recommend": рекомендации.stats, "llm": шлюз.stats()},
        errors=шлюз.ошибки())


def сохранить(итог: ИтогАнализа, папка: str | Path) -> Path:
    """Сохраняет результаты в файлах тех же форматов, что у скриптов шагов 3–12."""
    папка = Path(папка)
    папка.mkdir(parents=True, exist_ok=True)
    итог.taxonomy.сохранить(папка / "taxonomy.json")
    (папка / "mentions.json").write_text(
        json.dumps([asdict(м) for м in итог.mentions], ensure_ascii=False, indent=1), encoding="utf-8")
    сохранить_csv(итог.aggregate, папка / "aggregate.csv")
    (папка / "issues.json").write_text(
        json.dumps([asdict(п) for п in итог.problems], ensure_ascii=False, indent=1), encoding="utf-8")
    рекомендации, приоритеты = словари_для_отчёта(
        итог.recommendations, итог.prioritized, итог.ungrounded, доли_негатива(итог.aggregate, итог.product))
    for имя, данные in (("recommendations.json", рекомендации), ("prioritized.json", приоритеты)):
        (папка / имя).write_text(json.dumps(данные, ensure_ascii=False, indent=1), encoding="utf-8")
    (папка / "report.md").write_text(в_markdown(итог.report), encoding="utf-8")
    в_docx(итог.report, папка / "report.docx")
    return папка


def открыть_сохранённое(отзывы: list[Отзыв], *, товар: str, проблемы: list[dict],
                        агрегат: list[Характеристика], рекомендации: dict | None = None) -> Отчёт:
    """Отчёт по сохранённым файлам шагов 5, 6 и 8–9, без модели.

    Шаги 10–11 пересчитываются здесь же: они быстрые и не требуют модели, поэтому
    prioritized.json не нужен.
    """
    from review_agent.pipeline.recommend import Рекомендация

    предобработка = предобработать(отзывы)
    свои = [о for о in предобработка.reviews if о.is_own and о.product == товар]
    объекты = [Проблема(**п) for п in проблемы]
    рекомендации = рекомендации or {}
    принятые = [Рекомендация(**{к: в for к, в in п["recommendation"].items() if к != "kind"})
                for п in рекомендации.get("accepted", [])]
    доли = {х.attribute: х.negative_share for х in агрегат}
    прошли, отказы = проверить_опору(принятые, объекты, предобработка.reviews)
    приоритеты = расставить(прошли, объекты, доли_негатива=доли)
    без_доли = sorted({п.recommendation.attribute for п in приоритеты if not п.negative_share_known})
    return собрать(товар=товар, отзывов=len(свои), подозрительных=len(предобработка.suspicious_ids),
                   характеристики=агрегат, проблемы=проблемы,
                   приоритеты={"stats": {"negative_share_missing": без_доли},
                               "prioritized": [asdict(п) for п in приоритеты],
                               "ungrounded": [asdict(о) for о in отказы]},
                   рекомендации=рекомендации)
