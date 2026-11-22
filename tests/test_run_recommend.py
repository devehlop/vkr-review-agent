"""Проверка scripts/run_recommend.py с заглушкой вместо GigaChat."""
import importlib.util
import json
from pathlib import Path

from review_agent.llm.base import FakeLLMClient
from review_agent.pipeline.taxonomy import Атрибут, Справочник


def загрузить_скрипт():
    путь = Path(__file__).resolve().parent.parent / "scripts" / "run_recommend.py"
    спец = importlib.util.spec_from_file_location("run_recommend", путь)
    модуль = importlib.util.module_from_spec(спец)
    спец.loader.exec_module(модуль)
    return модуль


def test_скрипт_сохраняет_принятую_рекомендацию(tmp_path, capsys):
    csv = tmp_path / "o.csv"
    csv.write_text("id,Недостатки\n1,Молния заедает\n2,Молния ломается\n3,Молния заедает сильно\n"
                   "4,Всё отлично\n", encoding="utf-8")
    Справочник("одежда", [Атрибут("Фурнитура", ["молния"])]).сохранить(tmp_path / "t.json")
    проблема = {"attribute": "Фурнитура", "title": "Молния заедает", "mentions": 3, "reviews": 3,
                "coverage": 0.75, "quotes": ["Молния заедает", "Молния ломается"],
                "review_ids": ["1", "2", "3"], "title_ok": True}
    (tmp_path / "i.json").write_text(json.dumps([проблема], ensure_ascii=False), encoding="utf-8")
    (tmp_path / "a.csv").write_text(
        "product,is_own,attribute,mentions,reviews,coverage,positive,negative,neutral,"
        "negative_share,mean_rating,by_month\nКуртка,1,Фурнитура,3,3,0.75,0,3,0,1.000,,\n",
        encoding="utf-8")
    ответы = [
        json.dumps({"recommendations": [{"action": "Заменить молнию на металлическую YKK",
                    "type": "товар", "rationale": "75 % отзывов", "effort": 2,
                    "quotes": [{"n": 1, "text": "Молния заедает"}]}]}, ensure_ascii=False),
        json.dumps({"scores": {"specificity": 5, "grounding": 5, "feasibility": 4, "novelty": 5},
                    "comment": ""}),
    ]
    клиент = FakeLLMClient(ответы)
    выход = tmp_path / "r.json"
    код = загрузить_скрипт().main(
        [str(csv), "--product", "Куртка", "--taxonomy", str(tmp_path / "t.json"),
         "--issues", str(tmp_path / "i.json"), "--aggregate", str(tmp_path / "a.csv"),
         "--out", str(выход), "--count", "1", "--db", str(tmp_path / "c.sqlite")], клиент=клиент)
    assert код == 0
    данные = json.loads(выход.read_text(encoding="utf-8"))
    assert данные["accepted"][0]["recommendation"]["action"].startswith("Заменить молнию")
    assert "4 отзыв" not in клиент.calls[0][0].content and "из 4" in клиент.calls[0][0].content
    assert "100%" in клиент.calls[0][0].content          # доля негатива из aggregate.csv
    assert "Заменить молнию" in capsys.readouterr().out


def test_нет_файла_проблем(tmp_path, capsys):
    (tmp_path / "o.csv").write_text("id,text\n1,ок\n", encoding="utf-8")
    код = загрузить_скрипт().main([str(tmp_path / "o.csv"), "--product", "К", "--taxonomy",
                                   str(tmp_path / "нет.json")])
    assert код == 2 and "Файл не найден" in capsys.readouterr().err


def test_скрипт_с_сильными_сторонами(tmp_path, capsys):
    """С --mentions скрипт добавляет раздел strengths; тема с проблемой туда не попадает."""
    import json as j

    from review_agent.llm.base import FakeLLMClient as Заглушка
    from review_agent.pipeline.taxonomy import Атрибут, Справочник as С

    csv = tmp_path / "o.csv"
    csv.write_text("id,Достоинства,Недостатки\n1,Ткань мягкая,Молния заедает\n2,Ткань мягкая и тёплая,"
                   "Молния заедает сильно\n3,Ткань приятная,Молния заедает всегда\n", encoding="utf-8")
    С("одежда", [Атрибут("Материал", ["ткань"]), Атрибут("Фурнитура", ["молния"])]).сохранить(tmp_path / "t.json")
    упоминания = []
    for i, (плюс, минус) in enumerate([("Ткань мягкая", "Молния заедает"),
                                        ("Ткань мягкая и тёплая", "Молния заедает сильно"),
                                        ("Ткань приятная", "Молния заедает всегда")], 1):
        упоминания.append({"review_id": str(i), "sentence_idx": 0, "attribute": "Материал", "sentiment": "positive",
                           "source": "pros", "text": плюс, "product": "Куртка", "is_own": True})
        упоминания.append({"review_id": str(i), "sentence_idx": 1, "attribute": "Фурнитура", "sentiment": "positive",
                           "source": "cons", "text": минус, "product": "Куртка", "is_own": True})
    (tmp_path / "m.json").write_text(j.dumps(упоминания, ensure_ascii=False), encoding="utf-8")
    проблемы = [{"attribute": "Фурнитура", "title": "Молния заедает", "mentions": 3, "reviews": 3, "coverage": 1.0,
                 "quotes": ["Молния заедает"], "review_ids": ["1", "2", "3"], "title_ok": True}]
    (tmp_path / "i.json").write_text(j.dumps(проблемы, ensure_ascii=False), encoding="utf-8")

    def генератор(действие, цитата):
        return j.dumps({"recommendations": [{"action": действие, "type": "карточка товара",
                                             "rationale": "Следует из отзывов", "expected_effect": "Лучше",
                                             "effort": 1, "quotes": [{"n": 1, "text": цитата}]}]}, ensure_ascii=False)
    ок = j.dumps({"scores": {"specificity": 5, "grounding": 5, "feasibility": 5, "novelty": 5}, "comment": ""})
    ответы = [генератор("Заменить молнию на металлическую у поставщика", "Молния заедает"), ок,
              генератор("Добавить в заголовок «мягкая ткань»", "Ткань мягкая"), ок]
    выход = tmp_path / "r.json"
    код = загрузить_скрипт().main(
        [str(csv), "--product", "Куртка", "--taxonomy", str(tmp_path / "t.json"), "--issues", str(tmp_path / "i.json"),
         "--mentions", str(tmp_path / "m.json"), "--count", "1", "--out", str(выход), "--db", str(tmp_path / "c.sqlite")],
        клиент=Заглушка(ответы))
    assert код == 0
    итог = j.loads(выход.read_text(encoding="utf-8"))
    [сильная] = итог["strengths"]["accepted"]
    assert сильная["recommendation"]["attribute"] == "Материал" and сильная["recommendation"]["kind"] == "strength"
    assert "Сильные стороны" in capsys.readouterr().out
