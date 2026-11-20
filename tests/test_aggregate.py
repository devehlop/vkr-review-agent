"""Тесты агрегации: руками посчитанные эталонные значения."""
from datetime import date

import pytest

from review_agent.ingest.loader import Отзыв
from review_agent.pipeline.aggregate import агрегировать, сохранить_csv
from review_agent.pipeline.extract import Упоминание


def отзыв(id, rating, дата, product="Куртка", is_own=True):
    return Отзыв(id=id, product=product, is_own=is_own, text="т", rating=rating, date=дата)


def упом(review, attr, sent, product="Куртка", is_own=True):
    return Упоминание(review, 0, attr, sent, "text", "предложение", product, is_own)


ОТЗЫВЫ = [отзыв("1", 5, date(2026, 9, 10)), отзыв("2", 2, date(2026, 9, 20)),
          отзыв("3", 1, date(2026, 10, 5)), отзыв("4", 4, None),
          отзыв("к1", 3, date(2026, 10, 1), "Чужая", False)]
УПОМИНАНИЯ = [упом("1", "Размер", "positive"), упом("2", "Размер", "negative"),
              упом("2", "Размер", "negative"),                # второе упоминание в том же отзыве
              упом("3", "Размер", "negative"), упом("4", "Размер", "neutral"),
              упом("2", "Ткань", "negative"),
              упом("к1", "Размер", "positive", "Чужая", False)]


def найти(стат, товар, атрибут):
    return next(с for с in стат if с.product == товар and с.attribute == атрибут)


def test_счётчики_охват_и_доля_негатива():
    с = найти(агрегировать(ОТЗЫВЫ, УПОМИНАНИЯ), "Куртка", "Размер")
    assert (с.mentions, с.reviews) == (5, 4)             # 4 разных отзыва, 5 упоминаний
    assert (с.positive, с.negative, с.neutral) == (1, 3, 1)
    assert с.coverage == pytest.approx(4 / 4)             # у товара 4 отзыва
    assert с.negative_share == pytest.approx(3 / 5)


def test_средняя_оценка_по_отзывам_с_упоминанием():
    с = найти(агрегировать(ОТЗЫВЫ, УПОМИНАНИЯ), "Куртка", "Ткань")
    assert с.mean_rating == 2.0 and с.coverage == pytest.approx(1 / 4)
    assert найти(агрегировать(ОТЗЫВЫ, УПОМИНАНИЯ), "Куртка", "Размер").mean_rating == pytest.approx(3.0)


def test_динамика_по_месяцам_пропускает_отзывы_без_даты():
    с = найти(агрегировать(ОТЗЫВЫ, УПОМИНАНИЯ), "Куртка", "Размер")
    assert с.by_month == {"2026-09": (3, pytest.approx(2 / 3)), "2026-10": (1, 1.0)}


def test_товары_считаются_раздельно_свой_первым():
    стат = агрегировать(ОТЗЫВЫ, УПОМИНАНИЯ)
    assert [(с.product, с.attribute) for с in стат][-1] == ("Чужая", "Размер")
    чужой = найти(стат, "Чужая", "Размер")
    assert (чужой.mentions, чужой.coverage, чужой.is_own) == (1, 1.0, False)


def test_порядок_атрибутов_из_справочника():
    стат = агрегировать(ОТЗЫВЫ, УПОМИНАНИЯ, порядок_атрибутов=["Ткань", "Размер"])
    assert [с.attribute for с in стат if с.is_own] == ["Ткань", "Размер"]


def test_пустой_ввод():
    assert агрегировать([], []) == []


def test_csv(tmp_path):
    путь = tmp_path / "a.csv"
    сохранить_csv(агрегировать(ОТЗЫВЫ, УПОМИНАНИЯ), путь)
    строки = путь.read_text(encoding="utf-8").splitlines()
    assert строки[0].startswith("product,is_own,attribute") and len(строки) == 4
