"""Тесты загрузки отзывов на временных файлах."""
from datetime import date

import openpyxl
import pytest

from review_agent.ingest.loader import ЗагрузкаОшибка, загрузить, определить_колонки


def csv_файл(tmp_path, содержимое, кодировка="utf-8", имя="o.csv"):
    p = tmp_path / имя
    p.write_bytes(содержимое.encode(кодировка))
    return p


def test_автоопределение_колонок():
    м = определить_колонки(["ID", "Автор", "Отзыв", "Оценка", "Дата", "Достоинства", "Недостатки"])
    assert м == {"external_id": "ID", "text": "Отзыв", "rating": "Оценка", "date": "Дата",
                 "pros": "Достоинства", "cons": "Недостатки"}


def test_csv_с_точкой_с_запятой_и_cp1251(tmp_path):
    p = csv_файл(tmp_path, "Отзыв;Оценка;Дата\nХорошая куртка;5;2026-10-01\nМала;2,0;01.10.2026\n",
                 кодировка="cp1251")
    отзывы = загрузить(p, product="Куртка")
    assert [o.text for o in отзывы] == ["Хорошая куртка", "Мала"]
    assert [o.rating for o in отзывы] == [5.0, 2.0]
    assert [o.date for o in отзывы] == [date(2026, 10, 1)] * 2
    assert отзывы[0].is_own and отзывы[0].id == "o:1"


def test_персональные_данные_не_попадают_в_отзыв(tmp_path):
    p = csv_файл(tmp_path, "Отзыв,Автор,Фото\nНорм,Иван Петров,http://x/1.jpg\n")
    отзыв = загрузить(p, product="X")[0]
    assert "Иван" not in repr(отзыв) and "jpg" not in repr(отзыв)


def test_ручное_сопоставление_и_конкурент(tmp_path):
    p = csv_файл(tmp_path, "msg,stars\nмного брака,1\n")
    отзыв = загрузить(p, product="Конкурент А", is_own=False, колонки={"text": "msg", "rating": "stars"})[0]
    assert отзыв.text == "много брака" and отзыв.rating == 1.0 and not отзыв.is_own


def test_xlsx(tmp_path):
    книга = openpyxl.Workbook()
    ws = книга.active
    ws.append(["id", "Плюсы", "Минусы", "Рейтинг"])
    ws.append(["a1", "мягкая", "маломерит", 4])
    p = tmp_path / "o.xlsx"
    книга.save(p)
    o = загрузить(p, product="X")[0]
    assert (o.id, o.pros, o.cons, o.rating) == ("a1", "мягкая", "маломерит", 4.0)


def test_нет_колонки_с_текстом(tmp_path):
    p = csv_файл(tmp_path, "a,b\n1,2\n")
    with pytest.raises(ЗагрузкаОшибка):
        загрузить(p, product="X")


def test_неподдерживаемый_формат(tmp_path):
    p = tmp_path / "o.txt"
    p.write_text("x")
    with pytest.raises(ЗагрузкаОшибка):
        загрузить(p, product="X")
