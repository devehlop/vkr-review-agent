"""Модели данных предметной области (см. docs/architecture.md, раздел 6)."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

# Тональность упоминания: −1 негатив, 0 нейтрально, 1 позитив (FR-6)
Sentiment = Literal[-1, 0, 1]

# Из какого поля отзыва взято предложение (FR-2)
SourceField = Literal["comment", "pros", "cons"]


class Product(BaseModel):
    id: str
    name: str
    is_own: bool = True
    marketplace: str = ""


class Review(BaseModel):
    id: str
    product_id: str
    text: str = ""
    pros: str = ""
    cons: str = ""
    rating: int | None = Field(default=None, ge=1, le=5)
    created: date | None = None
    variant: str = ""
    suspicious: bool = False


class Sentence(BaseModel):
    id: str
    review_id: str
    text: str
    source_field: SourceField = "comment"


class TaxonomyItem(BaseModel):
    attribute: str
    feature: str


class Mention(BaseModel):
    id: str = ""
    sentence_id: str
    attribute: str
    feature: str
    sentiment: Sentiment
    quote: str


class AspectStats(BaseModel):
    """Итоги агрегации по атрибуту (и, если задан, по признаку) для одного товара."""

    product_id: str
    attribute: str
    feature: str | None = None
    mentions: int
    reviews_with: int
    # Охват: доля отзывов товара, где аспект упомянут
    coverage: float
    neg_share: float
    pos_share: float
    avg_rating: float | None = None


class Issue(BaseModel):
    id: str
    product_id: str
    attribute: str
    title: str
    mention_ids: list[str] = Field(default_factory=list)
    coverage: float = 0.0
    neg_share: float = 0.0


class RecommendationType(str, Enum):
    PRODUCT = "товар"
    PACKAGING = "упаковка"
    CARD = "карточка товара"
    SIZE_CHART = "размерная сетка"
    LOGISTICS = "логистика"


class Recommendation(BaseModel):
    id: str
    issue_id: str | None = None
    action: str
    type: RecommendationType
    rationale: str
    expected_effect: str = ""
    effort: int = Field(ge=1, le=3)
    quotes: list[str] = Field(default_factory=list)
    priority: float | None = None
    grounded: bool | None = None
