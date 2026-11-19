"""Клиент YandexGPT (Yandex Cloud AI Studio, Foundation Models API).

Не проверено на реальном API (ключей пока нет). Нужны API-ключ сервисного
аккаунта (YANDEX_API_KEY) и идентификатор каталога (YANDEX_FOLDER_ID).
"""

from __future__ import annotations

import os

import httpx

from review_agent.llm.base import LLMClient, LLMError, LLMResponse, Message

API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"


class YandexGPTClient(LLMClient):
    provider = "yandex"

    def __init__(
        self,
        model: str = "yandexgpt/latest",
        api_key: str | None = None,
        folder_id: str | None = None,
        api_url: str = API_URL,
        price_in_per_1k: float = 0.0,
        price_out_per_1k: float = 0.0,
        http: httpx.Client | None = None,
    ):
        super().__init__(model)
        self.api_key = api_key or os.getenv("YANDEX_API_KEY", "")
        self.folder_id = folder_id or os.getenv("YANDEX_FOLDER_ID", "")
        self.api_url = api_url
        self.price_in_per_1k = price_in_per_1k
        self.price_out_per_1k = price_out_per_1k
        self.http = http or httpx.Client(timeout=120)

    def complete(self, messages: list[Message], temperature: float = 0.0, max_tokens: int = 2048) -> LLMResponse:
        if not (self.api_key and self.folder_id):
            raise LLMError("Не заданы YANDEX_API_KEY и YANDEX_FOLDER_ID")
        payload = {
            "modelUri": f"gpt://{self.folder_id}/{self.model}",
            "completionOptions": {"stream": False, "temperature": temperature, "maxTokens": str(max_tokens)},
            "messages": [{"role": m.role, "text": m.content} for m in messages],
        }
        try:
            r = self.http.post(
                self.api_url,
                headers={"Authorization": f"Api-Key {self.api_key}", "x-folder-id": self.folder_id},
                json=payload,
            )
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMError(f"YandexGPT: {e}") from e
        result = r.json()["result"]
        usage = result.get("usage", {})
        return LLMResponse(
            text=result["alternatives"][0]["message"]["text"],
            tokens_in=int(usage.get("inputTextTokens", 0)),
            tokens_out=int(usage.get("completionTokens", 0)),
            model=f"{self.model}@{result.get('modelVersion', '')}".rstrip("@"),
        )
