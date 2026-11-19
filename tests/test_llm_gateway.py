import pytest
from pydantic import BaseModel

from review_agent.llm.base import LLMError, LLMResponse, Message
from review_agent.llm.fake import FakeLLMClient
from review_agent.llm.gateway import LLMFormatError, LLMGateway, extract_json


class Answer(BaseModel):
    value: int


MSGS = [Message("user", "Сколько?")]


def test_extract_json_variants():
    assert extract_json('```json\n{"value": 1}\n```') == {"value": 1}
    assert extract_json('Вот ответ: {"value": 2}. Готово') == {"value": 2}
    assert extract_json("[1, 2]") == [1, 2]
    with pytest.raises(ValueError):
        extract_json("нет json")


def test_structured_retries_with_error_feedback():
    client = FakeLLMClient(["не знаю", '{"value": "много"}', '{"value": 3}'])
    gw = LLMGateway(client)
    assert gw.structured("test", MSGS, Answer).value == 3
    # Во втором и третьем запросе модель видит свой ответ и текст ошибки
    assert len(client.calls) == 3
    assert client.calls[2][-1].role == "user" and "не прошёл проверку" in client.calls[2][-1].content
    assert [r.ok for r in gw.records] == [False, False, True]
    assert gw.summary()["test"]["calls"] == 3


def test_structured_custom_validator_and_failure():
    client = FakeLLMClient(['{"value": -1}'] * 3)
    gw = LLMGateway(client, max_format_retries=2)
    with pytest.raises(LLMFormatError) as e:
        gw.structured("test", MSGS, Answer, validate=lambda a: [] if a.value >= 0 else ["value < 0"])
    assert e.value.errors == ["value < 0"]
    assert len(gw.records) == 3


def test_cache_hits_are_free():
    client = FakeLLMClient(['{"value": 1}'])
    client.price_in_per_1k = 1.0
    gw = LLMGateway(client)
    gw.structured("s", MSGS, Answer)
    gw.structured("s", MSGS, Answer)
    assert len(client.calls) == 1
    assert gw.records[1].cached and gw.records[1].cost == 0
    assert gw.records[0].cost > 0


class FlakyClient(FakeLLMClient):
    def __init__(self):
        super().__init__(['{"value": 5}'])
        self.fails = 2

    def complete(self, messages, temperature=0.0, max_tokens=2048) -> LLMResponse:
        if self.fails:
            self.fails -= 1
            raise LLMError("429 Too Many Requests")
        return super().complete(messages, temperature, max_tokens)


def test_network_retries():
    sleeps = []
    gw = LLMGateway(FlakyClient(), sleep=sleeps.append)
    assert gw.structured("s", MSGS, Answer).value == 5
    assert sleeps == [2.0, 4.0]
    assert [r.ok for r in gw.records] == [False, False, True]
