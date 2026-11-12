"""Тесты LLM-шлюза: проверка формата, повторы, кэш, журнал (NFR-3…NFR-6)."""
import json

import pytest
from pydantic import BaseModel

from review_agent.llm import FakeClient, FormatError, LLMError, LLMGateway


class Answer(BaseModel):
    value: int


MESSAGES = [{"role": "user", "content": "Сколько будет 2+2?"}]


def test_json_in_code_fence_is_accepted():
    gateway = LLMGateway(FakeClient(['Вот ответ:\n```json\n{"value": 4}\n```']))
    assert gateway.ask_json("тест", MESSAGES, Answer).value == 4
    assert gateway.usage.calls == 1 and gateway.usage.format_errors == 0


def test_bad_format_is_retried_with_error_text():
    client = FakeClient(["не знаю", '{"value": "четыре"}', '{"value": 4}'])
    gateway = LLMGateway(client)
    assert gateway.ask_json("тест", MESSAGES, Answer).value == 4
    assert gateway.usage.format_errors == 2
    # в повторном запросе модель видит свой ответ и замечания
    assert client.calls[1][-2]["content"] == "не знаю"
    assert "Ответ не принят" in client.calls[1][-1]["content"]


def test_semantic_check_and_give_up():
    gateway = LLMGateway(FakeClient(['{"value": 5}']), format_retries=1)
    with pytest.raises(FormatError, match="должно быть 4"):
        gateway.ask_json("тест", MESSAGES, Answer, check=lambda a: [] if a.value == 4 else ["должно быть 4"])
    assert gateway.usage.calls == 2 and gateway.usage.failed == 1


def test_network_error_is_retried():
    client = FakeClient([LLMError("таймаут"), '{"value": 4}'])
    gateway = LLMGateway(client, retry_pause=0)
    assert gateway.ask_json("тест", MESSAGES, Answer).value == 4
    assert len(client.calls) == 2


def test_network_error_gives_up():
    gateway = LLMGateway(FakeClient([LLMError("нет сети")]), network_retries=2, retry_pause=0)
    with pytest.raises(LLMError):
        gateway.ask_json("тест", MESSAGES, Answer)
    assert gateway.usage.failed == 1


def test_cache_and_journal(tmp_path):
    client = FakeClient(['{"value": 4}'])
    client.price_in, client.price_out = 1.0, 2.0
    gateway = LLMGateway(client, journal_path=tmp_path / "logs/журнал.jsonl", cache_dir=tmp_path / "кэш")
    gateway.ask_json("тест", MESSAGES, Answer)
    gateway.ask_json("тест", MESSAGES, Answer)
    assert len(client.calls) == 1 and gateway.usage.cached == 1
    record = json.loads((tmp_path / "logs/журнал.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["step"] == "тест" and record["response"] == '{"value": 4}'
    expected = record["tokens_in"] / 1000 * 1.0 + record["tokens_out"] / 1000 * 2.0
    assert gateway.usage.cost == pytest.approx(expected)
