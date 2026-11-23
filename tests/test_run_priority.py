"""Проверка scripts/run_priority.py: проверка опоры и приоритеты по файлам шагов 5, 6 и 8."""
import importlib.util
import json
from pathlib import Path


def загрузить_скрипт():
    путь = Path(__file__).resolve().parent.parent / "scripts" / "run_priority.py"
    спец = importlib.util.spec_from_file_location("run_priority", путь)
    модуль = importlib.util.module_from_spec(спец)
    спец.loader.exec_module(модуль)
    return модуль


def принятая(проблема, действие, сложность, цитата):
    return {"recommendation": {
        "problem": проблема["title"], "attribute": проблема["attribute"], "action": действие,
        "type": "товар", "rationale": "жалобы", "expected_effect": "", "effort": сложность,
        "quotes": [цитата], "review_ids": проблема["review_ids"], "dropped_quotes": 0},
        "accepted": True, "verdicts": [], "reason": ""}


def test_скрипт_проверяет_опору_и_сортирует(tmp_path, capsys):
    (tmp_path / "o.csv").write_text(
        "id,Недостатки\n1,Молния заедает\n2,Молния ломается\n3,Молния заедает сильно\n"
        "4,Коробка помята\n5,Коробка помята и мокрая\n6,Коробка помята совсем\n7,Всё отлично\n",
        encoding="utf-8")
    молния = {"attribute": "Фурнитура", "title": "Молния заедает", "mentions": 3, "reviews": 3,
              "coverage": 3 / 7, "quotes": ["Молния заедает", "Молния ломается"],
              "review_ids": ["1", "2", "3"], "title_ok": True}
    коробка = {"attribute": "Упаковка", "title": "Мятая коробка", "mentions": 3, "reviews": 3,
               "coverage": 3 / 7, "quotes": ["Коробка помята"], "review_ids": ["4", "5", "6"],
               "title_ok": True}
    (tmp_path / "i.json").write_text(json.dumps([молния, коробка], ensure_ascii=False), encoding="utf-8")
    (tmp_path / "r.json").write_text(json.dumps({"stats": {"critic": True}, "accepted": [
        принятая(молния, "Заменить молнию на металлическую", 3, "Молния заедает"),
        принятая(коробка, "Перейти на жёсткую коробку", 2, "Коробка помята"),
        принятая(коробка, "Выдуманное действие", 1, "Коробка горит"),
    ], "rejected": [], "problems_without_recommendation": []}, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "a.csv").write_text(
        "product,is_own,attribute,mentions,reviews,coverage,positive,negative,neutral,"
        "negative_share,mean_rating,by_month\nКуртка,1,Фурнитура,3,3,0.43,0,3,0,1.000,,\n",
        encoding="utf-8")
    выход = tmp_path / "p.json"
    код = загрузить_скрипт().main([
        str(tmp_path / "o.csv"), "--product", "Куртка", "--issues", str(tmp_path / "i.json"),
        "--recommendations", str(tmp_path / "r.json"), "--aggregate", str(tmp_path / "a.csv"),
        "--out", str(выход)])
    assert код == 0
    данные = json.loads(выход.read_text(encoding="utf-8"))
    assert данные["stats"]["grounded"] == 2 and данные["stats"]["ungrounded"] == 1
    assert [п["recommendation"]["action"] for п in данные["prioritized"]] == [
        "Перейти на жёсткую коробку", "Заменить молнию на металлическую"]
    assert данные["stats"]["negative_share_missing"] == ["Упаковка"]
    assert "Коробка горит" in данные["ungrounded"][0]["reason"]
    вывод = capsys.readouterr().out
    assert "передайте --aggregate" in вывод and "отклонена: [Мятая коробка] Выдуманное действие" in вывод


def test_скрипт_сообщает_об_отсутствии_файла(tmp_path, capsys):
    код = загрузить_скрипт().main([str(tmp_path / "нет.csv"), "--product", "Куртка"])
    assert код == 2 and "Файл не найден" in capsys.readouterr().err
