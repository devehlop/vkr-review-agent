"""Шаг 3 конвейера: справочник «атрибут → признаки» по выборке отзывов (FR-7)."""
import random

from pydantic import BaseModel

from review_agent.config import load_prompt
from review_agent.llm.gateway import LLMGateway
from review_agent.models import Review

PROMPT_VERSION = "taxonomy_v1"


class Attribute(BaseModel):
    name: str
    description: str = ""
    features: list[str] = []


class Taxonomy(BaseModel):
    attributes: list[Attribute]

    def as_hints(self) -> dict[str, str]:
        """Формат для шага извлечения: название атрибута -> пояснение с признаками."""
        return {
            a.name: a.description + (f" (например: {', '.join(a.features)})" if a.features else "")
            for a in self.attributes
        }


def review_text(review: Review) -> str:
    """Отзыв одной строкой: достоинства, недостатки и комментарий вместе."""
    parts = [("Достоинства", review.pros), ("Недостатки", review.cons), ("Комментарий", review.text)]
    filled = [(label, text) for label, text in parts if text]
    if len(filled) == 1 and filled[0][0] == "Комментарий":
        return filled[0][1]
    return " ".join(f"{label}: {text}" for label, text in filled)


def build_taxonomy(
    gateway: LLMGateway,
    reviews: list[Review],
    category: str = "одежда",
    sample_size: int = 150,
    seed: int = 42,
    min_attributes: int = 5,
    max_attributes: int = 12,
    max_chars: int = 500,
) -> Taxonomy:
    """Просит модель составить справочник по случайной выборке отзывов.

    Выборка берётся из всех товаров сразу: один справочник нужен и для своего товара,
    и для конкурентов, иначе сравнение невозможно.
    """
    sample = random.Random(seed).sample(reviews, min(sample_size, len(reviews)))
    prompt = load_prompt(PROMPT_VERSION).format(
        category=category,
        min_attributes=min_attributes,
        max_attributes=max_attributes,
        reviews="\n".join(f"- {review_text(r)[:max_chars]}" for r in sample),
    )

    def check(result: Taxonomy) -> list[str]:
        names = [a.name.strip() for a in result.attributes]
        problems = []
        if not min_attributes <= len(names) <= max_attributes:
            problems.append(f"атрибутов должно быть от {min_attributes} до {max_attributes}, получено {len(names)}")
        if any(not name for name in names):
            problems.append("у атрибута пустое название")
        if len({name.lower() for name in names}) != len(names):
            problems.append("названия атрибутов повторяются")
        return problems

    taxonomy = gateway.ask_json("taxonomy", [{"role": "user", "content": prompt}], Taxonomy, check=check, max_tokens=2500)
    for attribute in taxonomy.attributes:
        attribute.name = attribute.name.strip()
    return taxonomy
