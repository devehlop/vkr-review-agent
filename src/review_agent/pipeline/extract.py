"""Шаг 4. Извлечение аспектов и тональности (FR-5, FR-6).

Предложения отправляются пакетами вместе со справочником. Ответ модели
проверяется: каждое предложение есть в ответе ровно один раз, атрибут
и признак взяты из справочника, цитата — подстрока предложения. При ошибке
шлюз возвращает модели текст ошибки и переспрашивает.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from review_agent.llm.gateway import LLMFormatError, LLMGateway
from review_agent.models import Mention, Sentence, Sentiment
from review_agent.preprocess.text import normalize_for_compare
from review_agent.prompts import render_prompt
from review_agent.taxonomy import Taxonomy

OTHER_FEATURE = "другое"
PROMPT = ("extract_aspects", "v1")


class ExtractedMention(BaseModel):
    attribute: str
    feature: str = OTHER_FEATURE
    sentiment: Sentiment
    quote: str


class SentenceResult(BaseModel):
    id: str
    mentions: list[ExtractedMention] = Field(default_factory=list)


class ExtractionBatch(BaseModel):
    results: list[SentenceResult]


@dataclass
class FewShotExample:
    sentence: str
    mentions: list[ExtractedMention]


@dataclass
class ExtractionResult:
    mentions: list[Mention]
    # Предложения из пакетов, на которые модель так и не дала корректный ответ
    failed_sentence_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def quote_in_sentence(quote: str, sentence: str) -> bool:
    """Цитата совпадает с фрагментом предложения с точностью до регистра, ё и знаков."""
    q = normalize_for_compare(quote)
    return bool(q) and q in normalize_for_compare(sentence)


def make_validator(local_ids: dict[str, Sentence], taxonomy: Taxonomy):
    def validate(batch: ExtractionBatch) -> list[str]:
        errors: list[str] = []
        ids = [r.id for r in batch.results]
        missing = [i for i in local_ids if i not in ids]
        extra = [i for i in ids if i not in local_ids]
        dups = sorted({i for i in ids if ids.count(i) > 1})
        if missing:
            errors.append(f"нет ответа для предложений: {', '.join(missing)}")
        if extra:
            errors.append(f"лишние номера предложений: {', '.join(extra)}")
        if dups:
            errors.append(f"повторяются номера: {', '.join(dups)}")
        for r in batch.results:
            sent = local_ids.get(r.id)
            if sent is None:
                continue
            for m in r.mentions:
                if not taxonomy.has(m.attribute):
                    errors.append(f"предложение {r.id}: атрибута «{m.attribute}» нет в справочнике")
                elif m.feature != OTHER_FEATURE and not taxonomy.has(m.attribute, m.feature):
                    errors.append(f"предложение {r.id}: у атрибута «{m.attribute}» нет признака «{m.feature}»")
                if not quote_in_sentence(m.quote, sent.text):
                    errors.append(f"предложение {r.id}: цитаты «{m.quote}» нет в тексте предложения")
        return errors

    return validate


def format_examples(examples: list[FewShotExample]) -> str:
    if not examples:
        return ""
    lines = ["Примеры:"]
    for ex in examples:
        answer = [m.model_dump() for m in ex.mentions]
        lines.append(f"Предложение: {ex.sentence}\nУпоминания: {json.dumps(answer, ensure_ascii=False)}")
    return "\n".join(lines) + "\n"


def build_messages(batch: list[Sentence], taxonomy: Taxonomy, examples: list[FewShotExample]):
    sentences = "\n".join(f"{i}. {s.text}" for i, s in enumerate(batch, start=1))
    return render_prompt(
        *PROMPT,
        taxonomy=taxonomy.as_prompt_text(),
        examples=format_examples(examples),
        sentences=sentences,
    )


def extract_mentions(
    gateway: LLMGateway,
    sentences: list[Sentence],
    taxonomy: Taxonomy,
    examples: list[FewShotExample] | None = None,
    batch_size: int = 25,
) -> ExtractionResult:
    """Прогоняет предложения пакетами. Неудачный пакет пропускается с отметкой (NFR-6)."""
    result = ExtractionResult(mentions=[])
    for start in range(0, len(sentences), batch_size):
        batch = sentences[start : start + batch_size]
        # Внутри пакета предложения нумеруются с 1: короткие номера экономят токены
        local = {str(i): s for i, s in enumerate(batch, start=1)}
        messages = build_messages(batch, taxonomy, examples or [])
        try:
            parsed = gateway.structured("extract", messages, ExtractionBatch, make_validator(local, taxonomy))
        except LLMFormatError as e:
            result.failed_sentence_ids.extend(s.id for s in batch)
            result.errors.append(str(e))
            continue
        for r in parsed.results:
            sent = local[r.id]
            for m in r.mentions:
                result.mentions.append(
                    Mention(
                        id=f"{sent.id}#{len(result.mentions)}",
                        sentence_id=sent.id,
                        attribute=m.attribute,
                        feature=m.feature,
                        sentiment=m.sentiment,
                        quote=m.quote,
                    )
                )
    return result


def mentions_to_labels(mentions: list[Mention], taxonomy: Taxonomy) -> dict[str, dict[str, int]]:
    """Переводит упоминания в формат эталонной разметки: {предложение: {код атрибута: тональность}}.

    Если атрибут упомянут в предложении несколько раз с разной тональностью:
    есть и −1, и 1 — считаем 0 (смешанная оценка), иначе берём ненулевую.
    """
    codes = {a.attribute: a.code for a in taxonomy.attributes}
    grouped: dict[str, dict[str, set[int]]] = {}
    for m in mentions:
        grouped.setdefault(m.sentence_id, {}).setdefault(codes[m.attribute], set()).add(m.sentiment)
    labels: dict[str, dict[str, int]] = {}
    for sid, attrs in grouped.items():
        labels[sid] = {}
        for code, values in attrs.items():
            if {-1, 1} <= values:
                labels[sid][code] = 0
            else:
                nonzero = values - {0}
                labels[sid][code] = nonzero.pop() if nonzero else 0
    return labels
