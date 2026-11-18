"""Тесты клиентов моделей на заглушке HTTP: настоящие серверы не вызываются."""
import json
import time

import httpx
import pytest

from review_agent.llm import LLMError
from review_agent.llm.gigachat import GigaChatClient
from review_agent.llm.openai_compat import OpenAICompatClient

REPLY = {"choices": [{"message": {"content": "привет"}}], "usage": {"prompt_tokens": 7, "completion_tokens": 3}}


def test_openai_compat_request_and_response():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["auth"], seen["body"] = str(request.url), request.headers["authorization"], json.loads(request.content)
        return httpx.Response(200, json=REPLY)

    client = OpenAICompatClient("yandex", "https://example.test/v1/", "gpt://папка/модель", api_key="key-1")
    client._http = httpx.Client(transport=httpx.MockTransport(handler))
    reply = client.complete([{"role": "user", "content": "тест"}], temperature=0.0, max_tokens=50)
    assert (reply.text, reply.tokens_in, reply.tokens_out) == ("привет", 7, 3)
    assert seen["url"] == "https://example.test/v1/chat/completions" and seen["auth"] == "Bearer key-1"
    assert seen["body"]["model"] == "gpt://папка/модель" and seen["body"]["max_tokens"] == 50


def test_http_error_becomes_llm_error():
    client = OpenAICompatClient("qwen", "http://localhost:1/v1", "m")
    client._http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(429, text="лимит")))
    with pytest.raises(LLMError, match="HTTP 429"):
        client.complete([{"role": "user", "content": "тест"}])


def test_gigachat_gets_token_once_and_uses_it():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.url.path, request.headers["authorization"]))
        if request.url.path.endswith("/oauth"):
            assert request.headers["rquid"] and b"scope=GIGACHAT_API_PERS" in request.content
            return httpx.Response(200, json={"access_token": "token123", "expires_at": (time.time() + 1800) * 1000})
        return httpx.Response(200, json=REPLY)

    client = GigaChatClient(auth_key="a2V5")
    client._http = httpx.Client(transport=httpx.MockTransport(handler))
    client.complete([{"role": "user", "content": "раз"}])
    client.complete([{"role": "user", "content": "два"}])
    assert calls == [
        ("/api/v2/oauth", "Basic a2V5"),
        ("/api/v1/chat/completions", "Bearer token123"),
        ("/api/v1/chat/completions", "Bearer token123"),
    ]
