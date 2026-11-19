"""Клиент GigaChat API (Сбер).

Схема авторизации: ключ авторизации (Basic) обменивается на токен доступа,
который живёт 30 минут, затем запросы идут с Bearer-токеном.

Не проверено на реальном API (ключей пока нет). Что проверить при первом
запуске: адрес получения токена, допустимость temperature близкой к нулю,
сертификат НУЦ Минцифры (путь к нему задаётся в GIGACHAT_CA_BUNDLE).
"""

from __future__ import annotations

import os
import time
import uuid

import httpx

from review_agent.llm.base import LLMClient, LLMError, LLMResponse, Message

AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
API_URL = "https://gigachat.devices.sberbank.ru/api/v1"

# Рублей за 1000 токенов по пакетным тарифам для физлиц (docs/architecture.md, раздел 4).
# Во Freemium запросы бесплатны, но стоимость считаем «как если бы платили»,
# чтобы сравнивать модели по NFR-3.
PRICES = {"GigaChat-2-Pro": 0.5}


class GigaChatClient(LLMClient):
    provider = "gigachat"

    def __init__(
        self,
        model: str = "GigaChat-2-Pro",
        auth_key: str | None = None,
        scope: str = "GIGACHAT_API_PERS",
        auth_url: str | None = None,
        api_url: str | None = None,
        verify: str | bool | None = None,
        http: httpx.Client | None = None,
    ):
        super().__init__(model)
        self.auth_key = auth_key or os.getenv("GIGACHAT_AUTH_KEY", "")
        self.scope = scope
        self.auth_url = auth_url or os.getenv("GIGACHAT_AUTH_URL", AUTH_URL)
        self.api_url = (api_url or os.getenv("GIGACHAT_API_URL", API_URL)).rstrip("/")
        if verify is None:
            verify = os.getenv("GIGACHAT_CA_BUNDLE") or True
        self.http = http or httpx.Client(timeout=120, verify=verify)
        self.price_in_per_1k = self.price_out_per_1k = PRICES.get(model, 0.0)
        self._token = ""
        self._token_expires = 0.0

    def _get_token(self) -> str:
        # Обновляем токен заранее, за минуту до истечения
        if self._token and time.time() < self._token_expires - 60:
            return self._token
        if not self.auth_key:
            raise LLMError("Не задан GIGACHAT_AUTH_KEY")
        try:
            r = self.http.post(
                self.auth_url,
                headers={
                    "Authorization": f"Basic {self.auth_key}",
                    "RqUID": str(uuid.uuid4()),
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
                data={"scope": self.scope},
            )
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMError(f"GigaChat: не удалось получить токен: {e}") from e
        data = r.json()
        self._token = data["access_token"]
        # expires_at приходит в миллисекундах
        self._token_expires = data.get("expires_at", 0) / 1000 or time.time() + 1800
        return self._token

    def complete(self, messages: list[Message], temperature: float = 0.0, max_tokens: int = 2048) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            # По документации temperature должна быть больше нуля
            "temperature": max(temperature, 0.01),
            "max_tokens": max_tokens,
        }
        try:
            r = self.http.post(
                f"{self.api_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._get_token()}", "Accept": "application/json"},
                json=payload,
            )
            if r.status_code == 401:
                # Токен мог истечь раньше срока: получаем новый и повторяем один раз
                self._token = ""
                r = self.http.post(
                    f"{self.api_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self._get_token()}", "Accept": "application/json"},
                    json=payload,
                )
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMError(f"GigaChat: {e}") from e
        data = r.json()
        usage = data.get("usage", {})
        return LLMResponse(
            text=data["choices"][0]["message"]["content"],
            tokens_in=int(usage.get("prompt_tokens", 0)),
            tokens_out=int(usage.get("completion_tokens", 0)),
            model=data.get("model", self.model),
        )
