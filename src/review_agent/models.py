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


@dataclass
class Issue:
    """Проблема товара: группа близких негативных упоминаний одного атрибута (шаг 6)."""

    id: int
    product: str
    attribute: str
    title: str
    sentence_ids: list[int]
    coverage: float = 0.0  # доля отзывов товара, в которых встречается проблема
    neg_share: float = 0.0  # доля негатива по атрибуту
    rarer_at_competitors: bool = False  # у конкурентов встречается заметно реже


@dataclass
class Recommendation:
    """Рекомендация продавцу (шаги 8–11)."""

    id: int
    issue_id: int
    action: str
    type: str  # товар, упаковка, карточка товара, размерная сетка, логистика
    rationale: str
    effort: int  # сложность исправления: 1 просто, 3 сложно
    quotes: list[tuple[int, str]] = field(default_factory=list)  # (номер предложения, цитата)
    priority: float = 0.0
    grounded: bool = False
