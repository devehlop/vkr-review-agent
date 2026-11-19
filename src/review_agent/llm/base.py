"""Общий интерфейс к языковым моделям (NFR-4).

Каждый провайдер реализует один метод complete(). Всё остальное (проверка
формата, повторы, кэш, журнал, стоимость) делает LLMGateway, поэтому
провайдеры получаются короткими и легко заменяются.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class LLMResponse:
    text: str
    tokens_in: int
    tokens_out: int
    model: str


class LLMError(RuntimeError):
    """Ошибка обращения к модели (сеть, авторизация, лимиты)."""


class LLMClient(ABC):
    #: Имя провайдера для журнала: gigachat, yandex, ollama, fake
    provider: str = ""
    #: Цены в рублях за 1000 токенов (NFR-3); 0 — бесплатно или неизвестно
    price_in_per_1k: float = 0.0
    price_out_per_1k: float = 0.0

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def complete(self, messages: list[Message], temperature: float = 0.0, max_tokens: int = 2048) -> LLMResponse:
        """Отправляет диалог модели и возвращает ответ."""

    def cost(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in * self.price_in_per_1k + tokens_out * self.price_out_per_1k) / 1000


def estimate_tokens(text: str) -> int:
    """Грубая оценка числа токенов для русского текста (около 3 символов на токен)."""
    return max(1, len(text) // 3)
