"""Клиент GigaChat API (REST).

Схема по документации Сбера:
1. POST {oauth_url} с заголовком Authorization: Basic <ключ авторизации>
   возвращает токен доступа на 30 минут;
2. POST {api_url}/v1/chat/completions с заголовком Authorization: Bearer <токен>.

Ключ авторизации берётся из переменной окружения GIGACHAT_CREDENTIALS
и нигде не сохраняется. Сертификат Минцифры задаётся через GIGACHAT_CA_BUNDLE.
"""
from __future__ import annotations

import os
import time
import uuid

import requests

from .base import LLMResponse, Message

OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
API_URL = "https://api.giga.chat"
# Токен живёт 30 минут; обновляем с запасом, чтобы не упасть посреди запроса.
ЗАПАС_СЕКУНД = 60


class GigaChatError(RuntimeError):
    """Ошибка обращения к GigaChat (без ключей и токенов в тексте)."""


class GigaChatClient:
    def __init__(self, credentials: str | None = None, *,
                 model: str = "GigaChat-2-Pro", scope: str = "GIGACHAT_API_PERS",
                 ca_bundle: str | None = None, timeout: float = 60.0,
                 structured_output: bool = False,
                 session: requests.Session | None = None):
        self._credentials = credentials or os.environ.get("GIGACHAT_CREDENTIALS", "")
        if not self._credentials:
            raise GigaChatError("Не задан ключ: переменная окружения GIGACHAT_CREDENTIALS")
        self.model = model
        self.scope = scope
        # verify=путь к сертификату Минцифры; проверку TLS не отключаем.
        self._verify = ca_bundle or os.environ.get("GIGACHAT_CA_BUNDLE") or True
        self._timeout = timeout
        # Строгий режим response_format=json_schema: сервер сам следит за валидностью JSON.
        # Выключен по умолчанию: поддержку схем нужно проверить на своей модели.
        self._structured = structured_output
        self._http = session or requests.Session()
        self._token = ""
        self._token_expires = 0.0  # время в секундах (time.time())

    def _получить_токен(self) -> str:
        if self._token and time.time() < self._token_expires - ЗАПАС_СЕКУНД:
            return self._token
        resp = self._http.post(
            OAUTH_URL,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "RqUID": str(uuid.uuid4()),
                "Authorization": f"Basic {self._credentials}",
            },
            data={"scope": self.scope},
            verify=self._verify, timeout=self._timeout,
        )
        if resp.status_code != 200:
            raise GigaChatError(f"Не удалось получить токен: HTTP {resp.status_code}")
        данные = resp.json()
        self._token = данные["access_token"]
        # В документации пример в секундах, в реальных ответах бывают миллисекунды:
        # значения больше 1e11 считаем миллисекундами.
        истекает = данные["expires_at"]
        self._token_expires = истекает / 1000 if истекает > 1e11 else float(истекает)
        return self._token

    def chat(self, messages: list[Message], *, temperature: float = 0.2,
             max_tokens: int = 1024, json_schema: dict | None = None) -> LLMResponse:
        тело = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_schema is not None and self._structured:
            тело["response_format"] = {"type": "json_schema", "schema": json_schema, "strict": True}
        resp = self._http.post(
            f"{API_URL}/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self._получить_токен()}",
            },
            json=тело,
            verify=self._verify, timeout=self._timeout,
        )
        if resp.status_code != 200:
            raise GigaChatError(f"Ошибка запроса к модели: HTTP {resp.status_code}")
        данные = resp.json()
        usage = данные.get("usage", {})
        return LLMResponse(
            text=данные["choices"][0]["message"]["content"],
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            model=данные.get("model", self.model),
        )
