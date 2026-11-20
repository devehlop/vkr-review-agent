"""Проверка scripts/run_aggregate.py на временных файлах."""
import importlib.util
import json
from pathlib import Path


def загрузить_скрипт():
    путь = Path(__file__).resolve().parent.parent / "scripts" / "run_aggregate.py"
    спец = importlib.util.spec_from_file_location("run_aggregate", путь)
    модуль = importlib.util.module_from_spec(спец)
    спец.loader.exec_module(модуль)
    return модуль


def test_скрипт_печатает_таблицу(tmp_path, capsys):
    csv = tmp_path / "o.csv"
    csv.write_text("id,Отзыв,Оценка\na,Маломерит,2\nb,Норм,5\n", encoding="utf-8")
    упоминание = {"review_id": "a", "sentence_idx": 0, "attribute": "Размер", "sentiment": "negative",
                  "source": "text", "text": "Маломерит", "product": "Куртка", "is_own": True}
    (tmp_path / "m.json").write_text(json.dumps([упоминание]), encoding="utf-8")
    код = загрузить_скрипт().main([str(csv), "--product", "Куртка", "--mentions", str(tmp_path / "m.json"),
                                   "--out", str(tmp_path / "a.csv")])
    вывод = capsys.readouterr().out
    assert код == 0 and "Размер" in вывод and "100%" in вывод and (tmp_path / "a.csv").exists()
