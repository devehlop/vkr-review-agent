"""Очистка отзывов: нормализация текста, пустые отзывы, точные дубликаты (FR-2)."""
import re
import unicodedata

from review_agent.models import CleanReport, Review

_SPACES = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Приводит текст к единой форме Юникода, убирает служебные символы и лишние пробелы."""
    text = unicodedata.normalize("NFC", text or "")
    # Категория C: управляющие и невидимые символы. Переводы строк заменяем пробелом.
    text = "".join(" " if ch in "\n\r\t" else ch for ch in text if ch in "\n\r\t" or unicodedata.category(ch)[0] != "C")
    return _SPACES.sub(" ", text).strip()


def clean_reviews(reviews: list[Review]) -> tuple[list[Review], CleanReport]:
    """Возвращает очищенные отзывы и отчёт о том, что было отброшено."""
    report = CleanReport(total=len(reviews))
    seen = set()
    kept = []
    for review in reviews:
        review.text = normalize_text(review.text)
        review.pros = normalize_text(review.pros)
        review.cons = normalize_text(review.cons)
        if not (review.text or review.pros or review.cons):
            report.empty += 1
            continue
        key = (review.product, review.text.lower(), review.pros.lower(), review.cons.lower())
        if key in seen:
            report.duplicates += 1
            continue
        seen.add(key)
        kept.append(review)
    report.kept = len(kept)
    return kept, report
