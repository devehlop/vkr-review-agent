"""Разбиение отзывов на предложения (FR-3)."""
from razdel import sentenize

from review_agent.models import Review, Sentence

# Поле отзыва -> метка источника. Предложение из «Недостатков» почти всегда негативное,
# поэтому источник передаётся модели как подсказка.
FIELDS = (("pros", "достоинства"), ("cons", "недостатки"), ("text", "комментарий"))


def split_text(text: str) -> list[str]:
    """Делит текст на предложения библиотекой razdel."""
    return [s.text.strip() for s in sentenize(text) if s.text.strip()]


def split_reviews(reviews: list[Review], min_chars: int = 3) -> list[Sentence]:
    """Возвращает предложения всех отзывов со сквозной нумерацией."""
    sentences = []
    for review in reviews:
        for attr, label in FIELDS:
            for text in split_text(getattr(review, attr)):
                if len(text) >= min_chars:
                    sentences.append(Sentence(len(sentences) + 1, review.id, text, label))
    return sentences
