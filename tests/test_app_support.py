"""Тесты вспомогательных функций веб-интерфейса."""
import os
from datetime import datetime

from review_agent import app_support as п


def test_папка_прогона_безопасное_имя():
    папка = п.папка_нового_прогона("Куртка «Зима» / 2026", datetime(2026, 11, 23, 9, 5))
    assert папка.parent == п.ПАПКА_ПРОГОНОВ and папка.name == "2026-11-23_09-05_Куртка_Зима_2026"
    assert п.папка_нового_прогона("///", datetime(2026, 1, 2, 3, 4)).name == "2026-01-02_03-04_товар"


def test_сохранить_загруженный():
    путь = п.сохранить_загруженный("id,Отзыв\n1,Хорошо\n".encode(), "../../выгрузка.csv")
    assert путь.name == "выгрузка.csv" and путь.read_text(encoding="utf-8").startswith("id,Отзыв")


def test_папки_с_результатами(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    for имя in ("issues.json", "aggregate.csv"):
        (tmp_path / "a" / имя).write_text("[]", encoding="utf-8")
    (tmp_path / "b" / "issues.json").write_text("[]", encoding="utf-8")   # нет aggregate.csv
    assert п.папки_с_результатами(tmp_path, tmp_path / "нет") == [tmp_path / "a"]


def test_ключ_задан(monkeypatch):
    monkeypatch.delenv("GIGACHAT_CREDENTIALS", raising=False)
    assert not п.ключ_задан()
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "x")
    assert п.ключ_задан() and os.environ["GIGACHAT_CREDENTIALS"] == "x"
