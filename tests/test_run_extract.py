"""Проверка scripts/run_extract.py с заглушкой вместо модели."""
import importlib.util
import json
from pathlib import Path

from review_agent.llm.base import FakeLLMClient
from review_agent.pipeline.taxonomy import Атрибут, Справочник


def загрузить_скрипт():
    путь = Path(__file__).resolve().parent.parent / "scripts" / "run_extract.py"
    спец = importlib.util.spec_from_file_location("run_extract", путь)
    модуль = importlib.util.module_from_spec(спец)
    спец.loader.exec_module(модуль)
    return модуль


def test_скрипт_извлекает_и_сохраняет(tmp_path, capsys):
    csv = tmp_path / "o.csv"
    csv.write_text("Недостатки,Достоинства\nМаломерит,Ткань мягкая\n", encoding="utf-8")
    Справочник("одежда", [Атрибут("Размер", ["маломерит"]), Атрибут("Ткань", ["мягкая"])]).сохранить(
        tmp_path / "t.json")
    ответ = json.dumps({"items": [
        {"i": 1, "mentions": [{"attribute": "Размер", "sentiment": "negative"}]},
        {"i": 2, "mentions": [{"attribute": "Ткань", "sentiment": "positive"}]}]}, ensure_ascii=False)
    выход = tmp_path / "m.json"
    код = загрузить_скрипт().main(
        [str(csv), "--product", "Куртка", "--taxonomy", str(tmp_path / "t.json"),
         "--out", str(выход), "--db", str(tmp_path / "c.sqlite")],
        клиент=FakeLLMClient([ответ]))
    assert код == 0
    м = json.loads(выход.read_text(encoding="utf-8"))
    assert {(x["attribute"], x["sentiment"]) for x in м} == {("Размер", "negative"), ("Ткань", "positive")}
    assert "Размер: +0 / -1 / =0" in capsys.readouterr().out
