"""Справочники аспектов (FR-7)."""

from __future__ import annotations

import json
from importlib import resources

from pydantic import BaseModel

from review_agent.models import TaxonomyItem


class TaxonomyAttribute(BaseModel):
    code: str
    attribute: str
    features: list[str]


class Taxonomy(BaseModel):
    version: str
    description: str = ""
    attributes: list[TaxonomyAttribute]

    def items(self) -> list[TaxonomyItem]:
        return [TaxonomyItem(attribute=a.attribute, feature=f) for a in self.attributes for f in a.features]

    def attribute_names(self) -> list[str]:
        return [a.attribute for a in self.attributes]

    def has(self, attribute: str, feature: str | None = None) -> bool:
        for a in self.attributes:
            if a.attribute == attribute:
                return feature is None or feature in a.features
        return False

    def as_prompt_text(self) -> str:
        """Справочник в виде списка для вставки в промпт."""
        return "\n".join(f"- {a.attribute}: {', '.join(a.features)}" for a in self.attributes)


def load_taxonomy(name: str = "clothing_v1") -> Taxonomy:
    """Загружает встроенный справочник по имени файла без расширения."""
    text = resources.files(__package__).joinpath(f"{name}.json").read_text(encoding="utf-8")
    return Taxonomy.model_validate(json.loads(text))
