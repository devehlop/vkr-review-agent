"""Фиктивный клиент для тестов и отладки без обращения к настоящей модели."""
from collections.abc import Callable

from .base import LLMResponse


class FakeClient:
    """Отдаёт заранее заданные ответы по очереди или вычисляет ответ функцией."""

    def __init__(self, responses: list[str] | Callable[[list[dict]], str], name: str = "fake"):
        self.name = name
        self.model = "fake-model"
        self.price_in = 0.0
        self.price_out = 0.0
        self.calls: list[list[dict]] = []
        self._responses = responses

    def complete(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 2000) -> LLMResponse:
        self.calls.append(messages)
        if callable(self._responses):
            text = self._responses(messages)
        else:
            text = self._responses[min(len(self.calls), len(self._responses)) - 1]
        if isinstance(text, Exception):
            raise text
        tokens_in = sum(len(m["content"]) for m in messages) // 4
        return LLMResponse(text=text, tokens_in=tokens_in, tokens_out=len(text) // 4)
