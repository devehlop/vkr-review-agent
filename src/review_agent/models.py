"""Основные структуры данных: отзыв и предложение."""
from dataclasses import dataclass, field
from datetime import date


@dataclass
class Review:
    """Отзыв покупателя. Имена и фотографии покупателей не загружаются (NFR-7)."""

    id: int
    product: str
    text: str = ""
    rating: int | None = None
    created: date | None = None
    pros: str = ""
    cons: str = ""
    variant: str = ""
    is_own: bool = True


@dataclass
class Sentence:
    """Предложение отзыва. source_field: из какого поля отзыва оно взято."""

    id: int
    review_id: int
    text: str
    source_field: str = "text"


@dataclass
class CleanReport:
    """Что сделала очистка: нужно для журнала и для главы 3."""

    total: int = 0
    empty: int = 0
    duplicates: int = 0
    kept: int = 0
    notes: list[str] = field(default_factory=list)
