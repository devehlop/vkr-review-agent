"""Тесты загрузки и предобработки (FR-1…FR-3)."""
import pandas as pd
import pytest

from review_agent.ingest import guess_mapping, load_reviews
from review_agent.models import Review
from review_agent.preprocess import clean_reviews, normalize_text, split_reviews


def test_guess_mapping_finds_russian_columns():
    mapping = guess_mapping(["Имя покупателя", "Оценка", "Текст отзыва", "Достоинства", "Дата"])
    assert mapping == {"text": "Текст отзыва", "rating": "Оценка", "created": "Дата", "pros": "Достоинства"}


def test_load_csv_with_semicolon(tmp_path):
    path = tmp_path / "отзывы.csv"
    path.write_text("Отзыв;Оценка;Дата;Имя\nХорошее платье;5;01.09.2026;Анна\nШвы кривые;2,0;;Пётр\n", encoding="utf-8")
    reviews = load_reviews(path, product="платье")
    assert [r.text for r in reviews] == ["Хорошее платье", "Швы кривые"]
    assert [r.rating for r in reviews] == [5, 2]
    assert reviews[0].created.isoformat() == "2026-09-01" and reviews[1].created is None
    # имя покупателя не попадает в данные
    assert not hasattr(reviews[0], "Имя")


def test_load_xlsx(tmp_path):
    path = tmp_path / "отзывы.xlsx"
    pd.DataFrame({"Достоинства": ["мягкая ткань"], "Недостатки": ["маломерит"], "Рейтинг": [4]}).to_excel(path, index=False)
    review = load_reviews(path, product="свитер", is_own=False)[0]
    assert (review.pros, review.cons, review.rating, review.is_own) == ("мягкая ткань", "маломерит", 4, False)


def test_load_without_text_column_fails(tmp_path):
    path = tmp_path / "плохой.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="не найдена колонка"):
        load_reviews(path, product="x")


def test_normalize_text_removes_service_symbols():
    assert normalize_text("  Хорошо​\n\nочень \t ") == "Хорошо очень"


def test_clean_removes_empty_and_duplicates():
    reviews = [
        Review(1, "p", "Отлично"),
        Review(2, "p", "  отлично "),
        Review(3, "p", ""),
        Review(4, "другой", "Отлично"),
    ]
    kept, report = clean_reviews(reviews)
    assert [r.id for r in kept] == [1, 4]
    assert (report.total, report.empty, report.duplicates, report.kept) == (4, 1, 1, 2)


def test_split_reviews_keeps_source_field():
    review = Review(7, "p", text="Ткань приятная. Но швы расходятся!", cons="Маломерит")
    sentences = split_reviews([review])
    assert [(s.text, s.source_field) for s in sentences] == [
        ("Маломерит", "недостатки"),
        ("Ткань приятная.", "комментарий"),
        ("Но швы расходятся!", "комментарий"),
    ]
    assert [s.id for s in sentences] == [1, 2, 3] and all(s.review_id == 7 for s in sentences)


def test_read_uploaded_file_and_manual_mapping():
    """Интерфейс передаёт загруженный файл как поток байтов и соответствие колонок, выбранное пользователем."""
    import io

    from review_agent.ingest import read_table, reviews_from_table

    data = "Покупатель;Мнение;Звёзд\nАнна;Хорошее платье;5\n".encode("utf-8")
    table = read_table(io.BytesIO(data), "выгрузка.csv")
    with pytest.raises(ValueError, match="не найдена колонка"):
        reviews_from_table(table, "платье")
    review = reviews_from_table(table, "платье", {"text": "Мнение", "rating": "Звёзд"}, is_own=False)[0]
    assert (review.text, review.rating, review.is_own) == ("Хорошее платье", 5, False)
