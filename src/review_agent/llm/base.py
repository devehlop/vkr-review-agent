"""Общий интерфейс клиента языковой модели и заглушка для тестов."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Message:
    """Одно сообщение диалога: роль (system/user/assistant) и текст."""
    role: str
    content: str


@dataclass
class LLMResponse:
    """Ответ модели с числом токенов (нужно для подсчёта стоимости)."""
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""


class LLMClient(Protocol):
    """Интерфейс, который реализуют GigaChat, YandexGPT, Ollama и заглушка."""

    def chat(self, messages: list[Message], *, temperature: float = 0.2,
             max_tokens: int = 1024) -> LLMResponse:
        ...


class FakeLLMClient:
    """Заглушка для тестов: возвращает заранее заданные ответы по очереди."""

    def __init__(self, answers: list[str]):
        self._answers = list(answers)
        self.calls: list[list[Message]] = []

    def chat(self, messages: list[Message], *, temperature: float = 0.2,
             max_tokens: int = 1024) -> LLMResponse:
        self.calls.append(messages)
        if not self._answers:
            raise RuntimeError("В заглушке закончились заготовленные ответы")
        return LLMResponse(text=self._answers.pop(0), model="fake")
