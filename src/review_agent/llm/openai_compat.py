"""Клиент для серверов с OpenAI-совместимым API: Yandex AI Studio, Ollama и другие."""
import httpx

from .base import LLMError, LLMResponse


class OpenAICompatClient:
    def __init__(
        self,
        name: str,
        base_url: str,
        model: str,
        api_key: str = "",
        price_in: float = 0.0,
        price_out: float = 0.0,
        timeout: float = 120.0,
        verify: bool | str = True,
    ):
        self.name = name
        self.model = model
        self.price_in = price_in
        self.price_out = price_out
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._embed_url = base_url.rstrip("/") + "/embeddings"
        self.embed_model = ""  # название модели эмбеддингов; пусто: эмбеддинги не настроены
        self._api_key = api_key
        self._http = httpx.Client(timeout=timeout, verify=verify)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def complete(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 2000) -> LLMResponse:
        payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        try:
            reply = self._http.post(self._url, json=payload, headers=self._headers())
            reply.raise_for_status()
            data = reply.json()
            usage = data.get("usage") or {}
            return LLMResponse(
                text=data["choices"][0]["message"]["content"] or "",
                tokens_in=int(usage.get("prompt_tokens", 0)),
                tokens_out=int(usage.get("completion_tokens", 0)),
            )
        except httpx.HTTPStatusError as error:
            raise LLMError(f"{self.name}: HTTP {error.response.status_code}: {error.response.text[:300]}") from error
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as error:
            raise LLMError(f"{self.name}: {type(error).__name__}: {error}") from error

    def embed(self, texts: list[str], batch_size: int = 50) -> list[list[float]]:
        """Векторные представления текстов (эмбеддинги) для группировки близких по смыслу жалоб."""
        if not self.embed_model:
            raise LLMError(f"{self.name}: модель эмбеддингов не задана")
        vectors: list[list[float]] = []
        try:
            for start in range(0, len(texts), batch_size):
                reply = self._http.post(
                    self._embed_url, headers=self._headers(),
                    json={"model": self.embed_model, "input": texts[start : start + batch_size]})
                reply.raise_for_status()
                items = sorted(reply.json()["data"], key=lambda item: item.get("index", 0))
                vectors.extend(item["embedding"] for item in items)
        except httpx.HTTPStatusError as error:
            raise LLMError(f"{self.name}: HTTP {error.response.status_code}: {error.response.text[:300]}") from error
        except (httpx.HTTPError, KeyError, ValueError) as error:
            raise LLMError(f"{self.name}: {type(error).__name__}: {error}") from error
        if len(vectors) != len(texts):
            raise LLMError(f"{self.name}: получено {len(vectors)} векторов на {len(texts)} текстов")
        return vectors
