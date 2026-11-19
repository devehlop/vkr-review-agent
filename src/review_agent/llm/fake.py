"""Модель-заглушка для разработки и тестов без ключей API.

Отвечает заранее заданными строками или функцией от диалога. Позволяет
прогнать весь конвейер, проверить повторы при неверном формате и журнал.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from review_agent.llm.base import LLMClient, LLMResponse, Message, estimate_tokens

Responder = Callable[[list[Message]], str]


class FakeLLMClient(LLMClient):
    provider = "fake"

    def __init__(self, responses: Iterable[str] | Responder, model: str = "fake-1"):
        super().__init__(model)
        if callable(responses):
            self._responder: Responder = responses
        else:
            queue = list(responses)

            def from_queue(_: list[Message]) -> str:
                if not queue:
                    raise AssertionError("FakeLLMClient: заготовленные ответы закончились")
                return queue.pop(0)

            self._responder = from_queue
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message], temperature: float = 0.0, max_tokens: int = 2048) -> LLMResponse:
        self.calls.append(list(messages))
        text = self._responder(messages)
        return LLMResponse(
            text=text,
            tokens_in=sum(estimate_tokens(m.content) for m in messages),
            tokens_out=estimate_tokens(text),
            model=self.model,
        )
