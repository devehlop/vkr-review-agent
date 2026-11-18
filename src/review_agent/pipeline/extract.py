"""Шаг 4 конвейера: извлечение аспектов и тональности по предложениям (FR-5, FR-6)."""
import json
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel

from review_agent.config import PROMPTS_DIR
from review_agent.llm.base import LLMError
from review_agent.llm.gateway import FormatError, LLMGateway
from review_agent.models import Sentence

PROMPT_VERSION = "extract_v1"


class Mention(BaseModel):
    """Одно упоминание характеристики товара в предложении."""

    attribute: str
    feature: str = ""
    sentiment: Literal[-1, 0, 1]
    quote: str


class SentenceResult(BaseModel):
    id: int
    mentions: list[Mention] = []


class BatchResult(BaseModel):
    results: list[SentenceResult]


@dataclass
class ExtractionResult:
    mentions: dict[int, list[Mention]] = field(default_factory=dict)  # номер предложения -> упоминания
    failed_ids: list[int] = field(default_factory=list)  # предложения, по которым ответ получить не удалось


def _norm(text: str) -> str:
    return " ".join(text.lower().replace("ё", "е").split())


def check_batch(result: BatchResult, batch: list[Sentence], attributes: list[str]) -> list[str]:
    """Смысловая проверка ответа: номера, атрибуты из справочника, цитата есть в предложении."""
    problems = []
    texts = {s.id: _norm(s.text) for s in batch}
    got = [r.id for r in result.results]
    if sorted(got) != sorted(texts):
        problems.append(f"номера предложений должны быть {sorted(texts)}, получены {sorted(got)}")
    for item in result.results:
        for mention in item.mentions:
            if mention.attribute not in attributes:
                problems.append(f"предложение {item.id}: атрибута «{mention.attribute}» нет в справочнике")
            if item.id in texts and _norm(mention.quote) not in texts[item.id]:
                problems.append(f"предложение {item.id}: цитаты «{mention.quote}» нет в тексте предложения")
    return problems


def build_messages(batch: list[Sentence], taxonomy: dict[str, str], category: str, examples: str = "") -> list[dict]:
    """Собирает запрос из файла промпта. taxonomy: название атрибута -> пояснение."""
    template = (PROMPTS_DIR / f"{PROMPT_VERSION}.md").read_text(encoding="utf-8")
    content = template.format(
        category=category,
        taxonomy="\n".join(f"- {name}: {hint}" for name, hint in taxonomy.items()),
        examples=f"\nПримеры разметки:\n{examples}\n" if examples else "",
        sentences="\n".join(
            json.dumps({"id": s.id, "источник": s.source_field, "текст": s.text}, ensure_ascii=False) for s in batch
        ),
    )
    return [{"role": "user", "content": content}]


def extract_mentions(
    gateway: LLMGateway,
    sentences: list[Sentence],
    taxonomy: dict[str, str],
    category: str = "одежда",
    batch_size: int = 20,
    examples: str = "",
) -> ExtractionResult:
    """Обрабатывает предложения пакетами. Пакет с неисправимым ответом пропускается с отметкой (NFR-6)."""
    output = ExtractionResult()
    attributes = list(taxonomy)
    for start in range(0, len(sentences), batch_size):
        batch = sentences[start : start + batch_size]
        try:
            result = gateway.ask_json(
                "extract",
                build_messages(batch, taxonomy, category, examples),
                BatchResult,
                check=lambda r, b=batch: check_batch(r, b, attributes),
                max_tokens=150 * len(batch) + 300,
            )
        except (FormatError, LLMError):
            output.failed_ids.extend(s.id for s in batch)
            continue
        for item in result.results:
            output.mentions[item.id] = item.mentions
    return output
