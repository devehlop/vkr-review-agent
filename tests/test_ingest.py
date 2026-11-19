from datetime import date

import pandas as pd

from review_agent.ingest.loader import ColumnMapping, guess_mapping, load_reviews, parse_date, parse_rating


def test_guess_mapping_russian_export():
    cols = ["Номер отзыва", "Дата публикации", "Количество звезд", "Достоинства", "Недостатки", "Комментарий", "Размер"]
    m = guess_mapping(cols)
    assert m == ColumnMapping(
        text="Комментарий", pros="Достоинства", cons="Недостатки",
        rating="Количество звезд", created="Дата публикации", variant="Размер", id="Номер отзыва",
    )


def test_guess_mapping_english():
    m = guess_mapping(["review", "sentiment"])
    assert m.text == "review" and m.rating is None


def test_parse_helpers():
    assert parse_rating("5") == 5
    assert parse_rating("4,0 из 5") == 4
    assert parse_rating("") is None
    assert parse_date("15.06.2026") == date(2026, 6, 15)
    assert parse_date("2026-06-15T10:00:00Z") == date(2026, 6, 15)


def test_load_csv_with_semicolon_and_bad_rows(tmp_path):
    p = tmp_path / "ozon.csv"
    p.write_text(
        "Дата;Оценка;Достоинства;Недостатки;Комментарий\n"
        "01.10.2026;5;Мягкая;;Отличный свитер\n"
        "02.10.2026;9;;;Плохо\n"
        "03.10.2026;;;;\n"
        "04.10.2026;2;;Маломерит;\n",
        encoding="utf-8",
    )
    r = load_reviews(p, product_id="own")
    assert [x.id for x in r.reviews] == ["own:2", "own:5"]
    assert r.reviews[0].pros == "Мягкая" and r.reviews[0].rating == 5
    assert r.reviews[1].cons == "Маломерит"
    assert [line for line, _ in r.skipped] == [3, 4]


def test_load_xlsx(tmp_path):
    p = tmp_path / "wb.xlsx"
    pd.DataFrame({"Текст отзыва": ["Ткань тонкая"], "Рейтинг": ["3"]}).to_excel(p, index=False)
    r = load_reviews(p, product_id="comp1")
    assert r.reviews[0].text == "Ткань тонкая" and r.reviews[0].rating == 3
