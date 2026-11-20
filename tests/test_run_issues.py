"""Проверка scripts/run_issues.py с заглушками вместо GigaChat."""
import importlib.util
import json
from pathlib import Path

from review_agent.llm.base import FakeLLMClient
from review_agent.llm.embeddings import FakeEmbedder
from review_agent.pipeline.taxonomy import Атрибут, Справочник


def загрузить_скрипт():
    путь = Path(__file__).resolve().parent.parent / "scripts" / "run_issues.py"
    спец = importlib.util.spec_from_file_location("run_issues", путь)
    модуль = importlib.util.module_from_spec(спец)
    спец.loader.exec_module(модуль)
    return модуль


def test_скрипт_находит_и_сохраняет_проблему(tmp_path, capsys):
    csv = tmp_path / "o.csv"
    csv.write_text("id,Недостатки\n1,Молния заедает при застёгивании\n2,Молния заедает после стирки\n"
                   "3,Молния заедает сильно\n", encoding="utf-8")
    Справочник("одежда", [Атрибут("Фурнитура", ["молния"])]).сохранить(tmp_path / "t.json")
    упоминания = [{"review_id": str(i), "sentence_idx": 0, "attribute": "Фурнитура", "sentiment": "negative",
                   "source": "cons", "text": т, "product": "Куртка", "is_own": True}
                  for i, т in ((1, "Молния заедает при застёгивании.") , (2, "Молния заедает после стирки."),
                               (3, "Молния заедает сильно."))]
    (tmp_path / "m.json").write_text(json.dumps(упоминания, ensure_ascii=False), encoding="utf-8")
    выход = tmp_path / "i.json"
    код = загрузить_скрипт().main(
        [str(csv), "--product", "Куртка", "--taxonomy", str(tmp_path / "t.json"),
         "--mentions", str(tmp_path / "m.json"), "--out", str(выход), "--threshold", "0.6",
         "--db", str(tmp_path / "c.sqlite")],
        клиент=FakeLLMClient(['{"title": "Молния заедает"}']), эмбеддер=FakeEmbedder())
    assert код == 0
    проблемы = json.loads(выход.read_text(encoding="utf-8"))
    assert проблемы[0]["title"] == "Молния заедает" and проблемы[0]["reviews"] == 3
    assert "Молния заедает" in capsys.readouterr().out
