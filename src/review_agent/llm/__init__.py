"""Доступ к языковым моделям. Модель выбирается настройкой (NFR-4)."""

from __future__ import annotations

from review_agent.config import Settings
from review_agent.llm.base import LLMClient, LLMError, LLMResponse, Message
from review_agent.llm.gateway import LLMCallRecord, LLMFormatError, LLMGateway

__all__ = ["LLMCallRecord", "LLMClient", "LLMError", "LLMFormatError", "LLMGateway", "LLMResponse", "Message", "make_client"]


def make_client(settings: Settings) -> LLMClient:
    """Создаёт клиента по имени провайдера из настроек."""
    provider = settings.llm_provider
    model = settings.llm_model
    if provider == "gigachat":
        from review_agent.llm.gigachat import GigaChatClient

        return GigaChatClient(model=model or "GigaChat-2-Pro")
    if provider == "yandex":
        from review_agent.llm.yandex import YandexGPTClient

        return YandexGPTClient(model=model or "yandexgpt/latest")
    if provider == "ollama":
        from review_agent.llm.ollama import OllamaClient

        return OllamaClient(model=model or "qwen3:4b")
    if provider == "fake":
        from review_agent.llm.fake import FakeLLMClient

        # Заглушка без заготовок отвечает пустым JSON: годится для пробного прогона
        return FakeLLMClient(lambda _: "{}", model=model or "fake-1")
    raise ValueError(f"Неизвестный провайдер LLM: {provider}")
