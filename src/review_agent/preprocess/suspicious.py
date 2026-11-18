"""Отметка подозрительных отзывов: почти одинаковые тексты у разных отзывов на один товар (FR-4).

Такие отзывы могут быть накруткой. Они не удаляются, а помечаются:
продавец видит их число и сам решает, учитывать ли их в анализе.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from review_agent.models import Review


def _full_text(review: Review) -> str:
    return " ".join(t for t in (review.pros, review.cons, review.text) if t).lower().replace("ё", "е")


def mark_suspicious(reviews: list[Review], threshold: float = 0.9, min_chars: int = 30, chunk: int = 500) -> int:
    """Помечает отзывы, текст которых почти совпадает с другим отзывом на тот же товар.

    Сходство: косинус между векторами TF-IDF по буквенным сочетаниям.
    Короткие отзывы («Всё отлично, спасибо») не проверяются: они совпадают и у честных покупателей.
    Точные дубликаты к этому моменту уже удалены очисткой. Возвращает число помеченных отзывов.
    """
    by_product: dict[str, list[Review]] = {}
    for review in reviews:
        review.suspicious = False
        if len(_full_text(review)) >= min_chars:
            by_product.setdefault(review.product, []).append(review)

    marked = 0
    for group in by_product.values():
        if len(group) < 2:
            continue
        # строки матрицы нормированы, поэтому произведение даёт косинус
        matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(_full_text(r) for r in group)
        for start in range(0, len(group), chunk):
            similarity = (matrix[start : start + chunk] @ matrix.T).toarray()
            rows = np.arange(similarity.shape[0])
            similarity[rows, rows + start] = 0.0  # сам с собой не сравниваем
            for offset in np.flatnonzero(similarity.max(axis=1) >= threshold):
                group[start + offset].suspicious = True
                marked += 1
    return marked
