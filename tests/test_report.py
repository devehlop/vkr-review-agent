"""Тесты шага 12: сборка отчёта и вывод в Markdown и DOCX. LLM не нужна."""
import importlib.util
import json
from datetime import date
from pathlib import Path

from docx import Document

from review_agent.report.build import Характеристика, собрать
from review_agent.report.render import в_docx, в_markdown

ПРОБЛЕМЫ = [
    {"attribute": "Упаковка", "title": "Мятая коробка", "mentions": 4, "reviews": 4, "coverage": 0.1,
     "quotes": ["Коробка помята", "коробка помята!", "Коробка порвана", "Пришла мятая", "Ещё одна"],
     "review_ids": ["1"]},
    {"attribute": "Размер", "title": "Маломерит", "mentions": 8, "reviews": 8, "coverage": 0.2,
     "quotes": ["Маломерит на размер"], "review_ids": ["2"]},
]
ХАРАКТЕРИСТИКИ = [
    Характеристика("Размер", 10, 0.25, 2, 8, 0, 0.8, 3.1),
    Характеристика("Упаковка", 5, 0.12, 1, 4, 0, 0.8, None),
    Характеристика("Ткань", 12, 0.3, 11, 1, 0, 0.083, 4.8),
    Характеристика("Цвет", 2, 0.05, 2, 0, 0, 0.0, 5.0),          # мало упоминаний: не сильная сторона
]


def приоритетная(действие, проблема, атрибут, приоритет, известна=True):
    return {"recommendation": {"problem": проблема, "attribute": атрибут, "action": действие,
                               "type": "карточка товара", "rationale": "жалобы в отзывах",
                               "expected_effect": "меньше возвратов", "effort": 1,
                               "quotes": ["Маломерит на размер"], "review_ids": ["2"]},
            "coverage": 0.2, "negative_share": 0.8 if известна else 1.0,
            "negative_share_known": известна, "behind_competitors": False, "priority": приоритет}


ПРИОРИТЕТЫ = {
    "stats": {"negative_share_missing": []},
    "prioritized": [приоритетная("Указать в карточке, что модель маломерит", "Маломерит", "Размер", 0.16)],
    "ungrounded": [{"problem": "Маломерит", "action": "Выдумка", "reason": "фрагмента «x» нет в цитатах проблемы"}],
}
РЕКОМЕНДАЦИИ = {
    "stats": {"critic": True},
    "rejected": [{"recommendation": {"problem": "Мятая коробка", "action": "Улучшить упаковку"},
                  "reason": "не прошла проверку после доработок"}],
    "strengths": {"accepted": [{"recommendation": {"problem": "Ткань",
                                                    "action": "Написать в описании, что ткань мягкая",
                                                    "quotes": ["ткань мягкая"]}}]},
}


def отчёт(**изменения):
    параметры = dict(товар="Куртка", отзывов=40, характеристики=ХАРАКТЕРИСТИКИ, проблемы=ПРОБЛЕМЫ,
                     приоритеты=ПРИОРИТЕТЫ, рекомендации=РЕКОМЕНДАЦИИ, дата=date(2026, 11, 23))
    параметры.update(изменения)
    return собрать(**параметры)


def test_сборка_отчёта():
    о = отчёт()
    assert [п.title for п in о.problems] == ["Маломерит", "Мятая коробка"]     # по охвату
    assert о.problems[0].has_recommendation and not о.problems[1].has_recommendation
    # повтор «коробка помята!» убран, остальные обрезаны до трёх
    assert о.problems[1].quotes == ["Коробка помята", "Коробка порвана", "Пришла мятая"]
    assert [с.attribute for с in о.strengths] == ["Ткань"]                     # без проблемных и редких
    assert о.strengths[0].card_tip == "Написать в описании, что ткань мягкая"
    assert [причина.split(":")[0] for _, _, причина in о.rejected] == ["критик", "нет опоры на отзывы"]
    assert о.warnings == [] and о.critic is True


def test_предупреждения():
    приоритеты = {**ПРИОРИТЕТЫ, "stats": {"negative_share_missing": ["Размер"]},
                  "prioritized": [приоритетная("Действие", "Маломерит", "Размер", 0.2, известна=False)]}
    о = отчёт(отзывов=12, подозрительных=2, приоритеты=приоритеты,
              рекомендации={**РЕКОМЕНДАЦИИ, "stats": {"critic": False}})
    текст = " ".join(о.warnings)
    assert "Отзывов всего 12" in текст and "Подозрительных отзывов" in текст
    assert "Размер не было доли негатива" in текст and "не проверялись критиком" in текст
    assert о.recommendations[0].negative_share is None


def test_markdown():
    md = в_markdown(отчёт())
    assert md.startswith("# Отчёт по отзывам: Куртка\n\nДата: 23.11.2026")
    assert "- Первое, что стоит сделать: Указать в карточке, что модель маломерит" in md
    assert "### 1. Указать в карточке, что модель маломерит" in md
    assert "Приоритет: 0.1600 (доля негатива 80 %)" in md
    assert "Мятая коробка\n\nУпаковка: 4 отзывов (10 %). Рекомендация не получена" in md
    assert "  - Для карточки товара: Написать в описании, что ткань мягкая" in md
    assert "| Ткань | 12 | 30 % | 11 | 1 | 8 % | 4.80 |" in md
    assert "- [Маломерит] Выдумка — нет опоры на отзывы" in md


def test_пустой_отчёт():
    md = в_markdown(отчёт(проблемы=[], приоритеты={}, рекомендации=None, характеристики=[]))
    assert "Рекомендаций, прошедших проверку, нет." in md and "Повторяющихся проблем не найдено." in md
    assert "Не выявлены." in md and "## Характеристики" not in md and "отклонённые" not in md


def test_docx(tmp_path):
    путь = в_docx(отчёт(), tmp_path / "папка" / "отчёт.docx")
    документ = Document(путь)
    текст = "\n".join(п.text for п in документ.paragraphs)
    assert "Отчёт по отзывам: Куртка" in текст
    assert "1. Указать в карточке, что модель маломерит" in текст
    assert "«Маломерит на размер»" in текст and "Для карточки товара:" in текст
    таблица = документ.tables[0]
    assert [я.text for я in таблица.rows[0].cells][:3] == ["Характеристика", "Упом.", "Охват"]
    assert len(таблица.rows) == 1 + len(ХАРАКТЕРИСТИКИ)


def test_скрипт_отчёта(tmp_path, capsys):
    путь = Path(__file__).resolve().parent.parent / "scripts" / "run_report.py"
    спец = importlib.util.spec_from_file_location("run_report", путь)
    скрипт = importlib.util.module_from_spec(спец)
    спец.loader.exec_module(скрипт)

    (tmp_path / "o.csv").write_text("id,Отзыв\n1,Маломерит на размер\n2,Коробка помята\n", encoding="utf-8")
    (tmp_path / "a.csv").write_text(
        "product,is_own,attribute,mentions,reviews,coverage,positive,negative,neutral,"
        "negative_share,mean_rating,by_month\nКуртка,1,Размер,10,10,0.25,2,8,0,0.800,3.10,\n"
        "Конкурент,0,Размер,5,5,0.5,5,0,0,0.000,,\n", encoding="utf-8")
    (tmp_path / "i.json").write_text(json.dumps(ПРОБЛЕМЫ, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "p.json").write_text(json.dumps(ПРИОРИТЕТЫ, ensure_ascii=False), encoding="utf-8")
    код = скрипт.main([str(tmp_path / "o.csv"), "--product", "Куртка", "--aggregate", str(tmp_path / "a.csv"),
                       "--issues", str(tmp_path / "i.json"), "--prioritized", str(tmp_path / "p.json"),
                       "--out", str(tmp_path / "r")])
    assert код == 0
    md = (tmp_path / "r.md").read_text(encoding="utf-8")
    assert "| Размер | 10 |" in md and "Конкурент" not in md       # в таблице только свой товар
    assert (tmp_path / "r.docx").is_file()
    вывод = capsys.readouterr().out
    assert "Рекомендаций: 1" in вывод and "Отзывов всего 2" in вывод
    assert скрипт.main([str(tmp_path / "нет.csv"), "--product", "Куртка"]) == 2
