"""Клиент GigaChat. Формат чата совпадает с OpenAI, отличается авторизация:
по ключу сначала выдаётся временный токен доступа (живёт около 30 минут)."""
import time
import uuid

import httpx

from .base import LLMError
from .openai_compat import OpenAICompatClient

AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
API_URL = "https://gigachat.devices.sberbank.ru/api/v1"


class GigaChatClient(OpenAICompatClient):
    def __init__(self, auth_key: str, model: str = "GigaChat-2-Pro", scope: str = "GIGACHAT_API_PERS", **kwargs):
        super().__init__(name="gigachat", base_url=API_URL, model=model, **kwargs)
        self._auth_key = auth_key
        self._scope = scope
        self._token = ""
        self._expires_at = 0.0

    def _headers(self) -> dict:
        # Обновляем токен за минуту до истечения срока
        if time.time() > self._expires_at - 60:
            try:
                reply = self._http.post(
                    AUTH_URL,
                    data={"scope": self._scope},
                    headers={"Authorization": f"Basic {self._auth_key}", "RqUID": str(uuid.uuid4())},
                )
                reply.raise_for_status()
                data = reply.json()
                self._token = data["access_token"]
                self._expires_at = data["expires_at"] / 1000  # в ответе миллисекунды
            except (httpx.HTTPError, KeyError, ValueError) as error:
                raise LLMError(f"gigachat: не удалось получить токен: {error}") from error
        return {"Authorization": f"Bearer {self._token}"}
