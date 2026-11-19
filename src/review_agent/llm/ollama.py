"""Клиент локальной открытой модели через Ollama (ключ не нужен)."""

from __future__ import annotations

import os

import httpx

from review_agent.llm.base import LLMClient, LLMError, LLMResponse, Message


class OllamaClient(LLMClient):
    provider = "ollama"

    def __init__(self, model: str = "qwen3:4b", base_url: str | None = None, http: httpx.Client | None = None):
        super().__init__(model)
        self.base_url = (base_url or os.getenv("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        # Локальная модель на процессоре может отвечать минутами
        self.http = http or httpx.Client(timeout=600)

    def complete(self, messages: list[Message], temperature: float = 0.0, max_tokens: int = 2048) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens, "seed": 42},
        }
        try:
            r = self.http.post(f"{self.base_url}/api/chat", json=payload)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMError(f"Ollama: {e}") from e
        data = r.json()
        return LLMResponse(
            text=data["message"]["content"],
            tokens_in=int(data.get("prompt_eval_count", 0)),
            tokens_out=int(data.get("eval_count", 0)),
            model=data.get("model", self.model),
        )
