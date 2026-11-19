import json

import httpx
import pytest

from review_agent.config import Settings
from review_agent.llm import LLMError, Message, make_client
from review_agent.llm.gigachat import GigaChatClient
from review_agent.llm.ollama import OllamaClient
from review_agent.llm.yandex import YandexGPTClient

MSGS = [Message("system", "Ты аналитик"), Message("user", "Привет")]


def test_gigachat_token_then_chat():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/oauth"):
            assert request.headers["Authorization"] == "Basic KEY"
            assert "RqUID" in request.headers
            return httpx.Response(200, json={"access_token": "TOKEN", "expires_at": 9_999_999_999_000})
        assert request.headers["Authorization"] == "Bearer TOKEN"
        body = json.loads(request.content)
        assert body["model"] == "GigaChat-2-Pro" and body["messages"][1]["content"] == "Привет"
        assert body["temperature"] > 0
        return httpx.Response(200, json={
            "model": "GigaChat-2-Pro:2.0",
            "choices": [{"message": {"role": "assistant", "content": "Здравствуйте"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        })

    client = GigaChatClient(auth_key="KEY", http=httpx.Client(transport=httpx.MockTransport(handler)))
    r1 = client.complete(MSGS)
    client.complete(MSGS)
    assert r1.text == "Здравствуйте" and (r1.tokens_in, r1.tokens_out) == (12, 3)
    # Токен получен один раз и переиспользуется
    assert sum(req.url.path.endswith("/oauth") for req in seen) == 1
    assert client.cost(1000, 1000) == pytest.approx(1.0)


def test_gigachat_http_error_becomes_llm_error():
    client = GigaChatClient(auth_key="KEY", http=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))))
    with pytest.raises(LLMError):
        client.complete(MSGS)


def test_yandex_payload_and_parse():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Api-Key K"
        body = json.loads(request.content)
        assert body["modelUri"] == "gpt://F/yandexgpt/latest"
        assert body["messages"][0] == {"role": "system", "text": "Ты аналитик"}
        return httpx.Response(200, json={"result": {
            "alternatives": [{"message": {"role": "assistant", "text": "Ок"}, "status": "ALTERNATIVE_STATUS_FINAL"}],
            "usage": {"inputTextTokens": "20", "completionTokens": "2", "totalTokens": "22"},
            "modelVersion": "5.1",
        }})

    client = YandexGPTClient(api_key="K", folder_id="F", http=httpx.Client(transport=httpx.MockTransport(handler)))
    r = client.complete(MSGS)
    assert (r.text, r.tokens_in, r.tokens_out) == ("Ок", 20, 2)


def test_yandex_without_keys():
    with pytest.raises(LLMError):
        YandexGPTClient(api_key="", folder_id="").complete(MSGS)


def test_ollama_parse():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        return httpx.Response(200, json={"model": "qwen3:4b", "message": {"content": "{}"}, "prompt_eval_count": 7, "eval_count": 1})

    r = OllamaClient(http=httpx.Client(transport=httpx.MockTransport(handler))).complete(MSGS)
    assert (r.text, r.tokens_in) == ("{}", 7)


def test_make_client():
    assert make_client(Settings(llm_provider="fake")).provider == "fake"
    assert make_client(Settings(llm_provider="ollama", llm_model="qwen3:8b")).model == "qwen3:8b"
    with pytest.raises(ValueError):
        make_client(Settings(llm_provider="unknown"))
