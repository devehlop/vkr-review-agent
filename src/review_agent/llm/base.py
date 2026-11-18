"""Общий интерфейс клиента LLM (NFR-4): модель меняется настройкой, а не кодом."""
from dataclasses import dataclass
from typing import Protocol


class LLMError(Exception):
    """Ошибка обращения к модели: сеть, авторизация, лимиты."""


@dataclass
class LLMResponse:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0


class LLMClient(Protocol):
    """Любой клиент модели: принимает сообщения чата, возвращает текст и число токенов."""

    name: str  # короткое имя для журнала и отчётов, например «gigachat»
    model: str  # точное название модели у провайдера
    price_in: float  # цена за 1000 токенов ввода
    price_out: float  # цена за 1000 токенов вывода

    def complete(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 2000) -> LLMResponse: ...
