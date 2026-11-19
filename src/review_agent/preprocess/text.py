"""Очистка текста, разбиение на предложения и дедупликация (FR-2, FR-3)."""

from __future__ import annotations

import re
import unicodedata

from razdel import sentenize

from review_agent.models import Review, Sentence, SourceField
from review_agent.preprocess.pii import mask_pii

_SPACES = re.compile(r"[ \t ​]+")
_MANY_NEWLINES = re.compile(r"\n{2,}")
# Повторы одного знака больше трёх раз: «!!!!!!» → «!!!», «ооочень» не трогаем
_REPEATED_PUNCT = re.compile(r"([!?.,)(])\1{3,}")
_CONTROL = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]")


def clean_text(text: str | None) -> str:
    """Нормализует текст: юникод, служебные символы, пробелы, повторы знаков."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    text = _REPEATED_PUNCT.sub(r"\1\1\1", text)
    text = _SPACES.sub(" ", text)
    text = _MANY_NEWLINES.sub("\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def normalize_for_compare(text: str) -> str:
    """Ключ для сравнения: нижний регистр, ё → е, только буквы и цифры."""
    text = text.lower().replace("ё", "е")
    return " ".join(re.findall(r"\w+", text))


def split_sentences(text: str) -> list[str]:
    """Делит текст на предложения библиотекой razdel.

    Переводы строк тоже считаются границей: покупатели часто пишут списком
    без точек.
    """
    result: list[str] = []
    for line in text.split("\n"):
        for sent in sentenize(line):
            s = sent.text.strip()
            # Предложение без единой буквы («!!!», «)))») анализировать нечего
            if s and re.search(r"[^\W\d_]", s):
                result.append(s)
    return result


def review_sentences(review: Review) -> list[Sentence]:
    """Склеивает поля отзыва и делит их на предложения с пометкой поля-источника."""
    fields: list[tuple[SourceField, str]] = [
        ("pros", review.pros),
        ("cons", review.cons),
        ("comment", review.text),
    ]
    sentences: list[Sentence] = []
    for source, raw in fields:
        for s in split_sentences(clean_text(raw)):
            sentences.append(
                Sentence(
                    id=f"{review.id}:{len(sentences)}",
                    review_id=review.id,
                    text=s,
                    source_field=source,
                )
            )
    return sentences


def prepare_review(review: Review) -> Review:
    """Чистит поля отзыва и маскирует персональные данные."""
    return review.model_copy(
        update={
            "text": mask_pii(clean_text(review.text)),
            "pros": mask_pii(clean_text(review.pros)),
            "cons": mask_pii(clean_text(review.cons)),
        }
    )


def is_empty(review: Review) -> bool:
    return not normalize_for_compare(f"{review.text} {review.pros} {review.cons}")


def deduplicate(reviews: list[Review]) -> tuple[list[Review], int]:
    """Убирает пустые отзывы и точные дубликаты внутри одного товара.

    Возвращает оставшиеся отзывы и число удалённых. Почти одинаковые тексты
    (возможные накрутки) здесь не удаляются: их помечает отдельный шаг по
    эмбеддингам.
    """
    seen: set[tuple[str, str]] = set()
    kept: list[Review] = []
    for r in reviews:
        if is_empty(r):
            continue
        key = (r.product_id, normalize_for_compare(f"{r.pros} | {r.cons} | {r.text}"))
        if key in seen:
            continue
        seen.add(key)
        kept.append(r)
    return kept, len(reviews) - len(kept)
