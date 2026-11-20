"""Проверка скрипта scripts/run_taxonomy.py с заглушкой вместо модели."""
import importlib.util
import json
from pathlib import Path

from review_agent.llm.base import FakeLLMClient
from review_agent.pipeline.taxonomy import Справочник

ОТВЕТ = json.dumps({"attributes": [
    {"name": "Размер", "features": ["маломерит"]}, {"name": "Ткань", "features": ["мягкая"]},
    {"name": "Доставка", "features": ["быстро"]}]}, ensure_ascii=False)


def загрузить_скрипт():
    путь = Path(__file__).resolve().parent.parent / "scripts" / "run_taxonomy.py"
    спец = importlib.util.spec_from_file_location("run_taxonomy", путь)
    модуль = importlib.util.module_from_spec(спец)
    спец.loader.exec_module(модуль)
    return модуль


def test_скрипт_строит_и_сохраняет_справочник(tmp_path, capsys):
    csv = tmp_path / "o.csv"
    csv.write_text("Отзыв,Оценка\nКуртка хорошая, но маломерит\nМягкая ткань, быстро доставили\n",
                   encoding="utf-8")
    выход = tmp_path / "t.json"
    код = загрузить_скрипт().main(
        [str(csv), "--product", "Куртка", "--category", "одежда",
         "--out", str(выход), "--db", str(tmp_path / "c.sqlite")],
        клиент=FakeLLMClient([ОТВЕТ]))
    assert код == 0
    assert [а.name for а in Справочник.загрузить(выход).attributes] == ["Размер", "Ткань", "Доставка"]
    assert "Справочник (3 атрибутов)" in capsys.readouterr().out


def test_пустой_файл_после_очистки(tmp_path):
    csv = tmp_path / "o.csv"
    csv.write_text("Отзыв\n \n", encoding="utf-8")
    assert загрузить_скрипт().main(
        [str(csv), "--product", "X", "--category", "Y", "--db", str(tmp_path / "c.sqlite")],
        клиент=FakeLLMClient([])) == 1


def test_несуществующий_файл_даёт_понятную_ошибку(tmp_path, capsys):
    код = загрузить_скрипт().main(
        [str(tmp_path / "нет.csv"), "--product", "X", "--category", "Y"],
        клиент=FakeLLMClient([]))
    assert код == 2 and "Файл не найден" in capsys.readouterr().err
