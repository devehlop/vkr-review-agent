"""Поиск и маскирование персональных данных в текстах отзывов (NFR-7).

Покупатели иногда оставляют в отзывах телефоны, почту, ссылки и номера заказов.
Такие фрагменты заменяются метками до сохранения в базу и до отправки в LLM.
Имена и адреса регулярными выражениями надёжно не находятся: это ограничение
фиксируется в тексте ВКР.
"""

from __future__ import annotations

import re

# Порядок важен: сначала почта и ссылки, потом телефоны, потом длинные числа
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("[EMAIL]", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("[ССЫЛКА]", re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)),
    # Российские номера: +7 / 8, затем 10 цифр с любыми разделителями
    (
        "[ТЕЛЕФОН]",
        re.compile(r"(?<!\d)(?:\+7|8)[\s\-()]*\d{3}[\s\-()]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}(?!\d)"),
    ),
    # Номера заказов, треков, карт: восемь и более цифр подряд
    ("[НОМЕР]", re.compile(r"(?<!\d)\d{8,}(?!\d)")),
    # Упоминания аккаунтов в соцсетях
    ("[АККАУНТ]", re.compile(r"(?<![\w.])@[A-Za-z0-9_]{3,}")),
]


def find_pii(text: str) -> list[str]:
    """Возвращает список меток найденных персональных данных (с повторами)."""
    found: list[str] = []
    for label, pattern in _PATTERNS:
        found.extend(label for _ in pattern.finditer(text))
        text = pattern.sub(label, text)
    return found


def has_pii(text: str) -> bool:
    return bool(find_pii(text))


def mask_pii(text: str) -> str:
    """Заменяет персональные данные метками вида [ТЕЛЕФОН]."""
    for label, pattern in _PATTERNS:
        text = pattern.sub(label, text)
    return text
