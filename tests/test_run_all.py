"""Полный прогон конвейера на заглушке модели: шаги 2–12, сохранение и повторное открытие."""
import json

from docx import Document

from review_agent.ingest.loader import Отзыв
from review_agent.llm.embeddings import FakeEmbedder
from review_agent.llm.gateway import LLMGateway
from review_agent.pipeline.run_all import открыть_сохранённое, запустить_анализ, сохранить
from review_agent.report.build import прочитать_агрегат
from review_agent.report.render import в_markdown
from tests.scripted_model import ScriptedLLMClient


def отзывы() -> list[Отзыв]:
    тексты = (["Молния заедает с первого дня"] * 0
              + [f"Молния заедает, покупка номер {i}" for i in range(5)]
              + [f"Сильно маломерит, заказ {i}" for i in range(4)]
              + [f"Очень тёплая куртка, носим {i} неделю" for i in range(8)]
              + ["Пришла мятая коробка", "Всё понравилось"])
    свои = [Отзыв(f"r{i}", "Куртка", True, text=т, rating=2 if "Молни" in т else 5) for i, т in enumerate(тексты)]
    чужие = [Отзыв(f"c{i}", "Конкурент", False, text=f"Молния хорошая, тёплая куртка {i}") for i in range(3)]
    return свои + чужие


def прогон(**параметры):
    модель = ScriptedLLMClient(**параметры)
    шаги = []
    итог = запустить_анализ(LLMGateway(модель), FakeEmbedder(), отзывы(), товар="Куртка",
                            категория="верхняя одежда", шаг=шаги.append)
    return итог, модель, шаги


def test_полный_прогон():
    итог, модель, шаги = прогон()
    assert шаги[0] == "Предобработка отзывов" and шаги[-1] == "Отчёт"
    assert модель.шаги[0] == "taxonomy" and "critic" in модель.шаги
    assert [п.attribute for п in итог.problems] == ["Фурнитура", "Размер и посадка"]
    assert итог.problems[0].reviews == 5 and итог.problems[0].coverage == 5 / 19   # только свои отзывы
    assert len(итог.prioritized) == 2 and итог.ungrounded == []
    отчёт = итог.report
    assert отчёт.reviews == 19 and [с.attribute for с in отчёт.strengths] == ["Материал"]
    assert отчёт.recommendations[0].problem == "Жалобы: фурнитура"
    assert итог.stats["llm"]["calls"] == len(модель.шаги)


def test_прогон_с_отклоняющим_критиком():
    итог, _, _ = прогон(оценка_критика=2)
    assert итог.prioritized == [] and len(итог.recommendations.rejected) == 2
    assert "Рекомендаций, прошедших проверку, нет." in в_markdown(итог.report)
    assert all("критик" in причина for _, _, причина in итог.report.rejected)


def test_сохранение_и_повторное_открытие(tmp_path):
    итог, _, _ = прогон()
    папка = сохранить(итог, tmp_path / "прогон")
    имена = {п.name for п in папка.iterdir()}
    assert {"taxonomy.json", "mentions.json", "aggregate.csv", "issues.json", "recommendations.json",
            "prioritized.json", "report.md", "report.docx"} <= имена
    assert "Жалобы: фурнитура" in "\n".join(п.text for п in Document(папка / "report.docx").paragraphs)

    # повторное открытие без модели: шаги 10–11 пересчитываются из сохранённых файлов
    повторно = открыть_сохранённое(
        отзывы(), товар="Куртка",
        проблемы=json.loads((папка / "issues.json").read_text(encoding="utf-8")),
        агрегат=прочитать_агрегат(папка / "aggregate.csv", "Куртка"),
        рекомендации=json.loads((папка / "recommendations.json").read_text(encoding="utf-8")))
    assert [р.action for р in повторно.recommendations] == [р.action for р in итог.report.recommendations]
    assert [п.title for п in повторно.problems] == [п.title for п in итог.report.problems]


def test_открытие_без_рекомендаций(tmp_path):
    итог, _, _ = прогон()
    папка = сохранить(итог, tmp_path)
    отчёт = открыть_сохранённое(отзывы(), товар="Куртка",
                                проблемы=json.loads((папка / "issues.json").read_text(encoding="utf-8")),
                                агрегат=прочитать_агрегат(папка / "aggregate.csv", "Куртка"))
    assert отчёт.recommendations == [] and len(отчёт.problems) == 2
