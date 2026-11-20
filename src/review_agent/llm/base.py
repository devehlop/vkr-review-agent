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
    finish_reason: str = ""   # stop = закончил сам; length = обрезан по лимиту токенов


class LLMClient(Protocol):
    """Интерфейс, который реализуют GigaChat, YandexGPT, Ollama и заглушка."""

    def chat(self, messages: list[Message], *, temperature: float = 0.2,
             max_tokens: int = 1024, json_schema: dict | None = None) -> LLMResponse:
        ...


class FakeLLMClient:
    """Заглушка для тестов: возвращает заранее заданные ответы по очереди."""

    def __init__(self, answers: list[str]):
        self._answers = list(answers)
        self.calls: list[list[Message]] = []
        self.temperatures: list[float] = []

    def chat(self, messages: list[Message], *, temperature: float = 0.2,
             max_tokens: int = 1024, json_schema: dict | None = None) -> LLMResponse:
        self.calls.append(messages)
        self.temperatures.append(temperature)
        if not self._answers:
            raise RuntimeError("В заглушке закончились заготовленные ответы")
        return LLMResponse(text=self._answers.pop(0), model="fake")
