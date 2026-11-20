"""Тесты LLM-шлюза: модель подменена заглушкой, сеть не нужна."""
import pytest

from review_agent.llm.base import FakeLLMClient, LLMResponse, Message
from review_agent.llm.gateway import LLMGateway, LLMGatewayError

СХЕМА = {"type": "object", "properties": {"аспект": {"type": "string"}},
         "required": ["аспект"]}
СООБЩЕНИЕ = [Message("user", "Выдели аспект")]


class Сбойный:
    """Первые n вызовов падают, потом отвечает."""
    model = "тест"

    def __init__(self, падений, ответ='{"аспект": "размер"}'):
        self.падений = падений
        self.ответ = ответ
        self.вызовов = 0

    def chat(self, messages, *, temperature=0.2, max_tokens=1024):
        self.вызовов += 1
        if self.вызовов <= self.падений:
            raise ConnectionError("сеть")
        return LLMResponse(self.ответ, prompt_tokens=100, completion_tokens=10, model="тест")


def шлюз(клиент, **kw):
    return LLMGateway(клиент, sleep=lambda _: None, **kw)


def test_разбор_json_в_блоке_кода():
    g = шлюз(FakeLLMClient(['```json\n{"аспект": "цвет"}\n```']))
    assert g.complete("аспекты", СООБЩЕНИЕ, schema=СХЕМА).data == {"аспект": "цвет"}


def test_повтор_после_сетевой_ошибки():
    клиент = Сбойный(падений=2)
    g = шлюз(клиент)
    assert g.complete("аспекты", СООБЩЕНИЕ, schema=СХЕМА).data["аспект"] == "размер"
    assert клиент.вызовов == 3
    assert g.stats()["errors"] == 2


def test_невалидный_ответ_по_схеме_повторяется_и_падает():
    g = шлюз(FakeLLMClient(['{"другое": 1}'] * 3), max_retries=3)
    with pytest.raises(LLMGatewayError):
        g.complete("аспекты", СООБЩЕНИЕ, schema=СХЕМА)
    assert g.stats()["errors"] == 3


def test_кэш_не_тратит_токены_повторно():
    клиент = Сбойный(падений=0)
    g = шлюз(клиент, price_per_1k_in=1.0, price_per_1k_out=2.0)
    первый = g.complete("аспекты", СООБЩЕНИЕ, schema=СХЕМА)
    второй = g.complete("аспекты", СООБЩЕНИЕ, schema=СХЕМА)
    assert not первый.from_cache and второй.from_cache
    assert клиент.вызовов == 1
    s = g.stats()
    assert s["calls"] == 2 and s["cache_hits"] == 1
    assert s["cost"] == pytest.approx(0.1 + 0.02)


def test_разные_параметры_разный_кэш():
    клиент = Сбойный(падений=0)
    g = шлюз(клиент)
    g.complete("а", СООБЩЕНИЕ, temperature=0.0)
    g.complete("а", СООБЩЕНИЕ, temperature=0.7)
    assert клиент.вызовов == 2


def test_без_схемы_возвращает_текст():
    g = шлюз(FakeLLMClient(["просто текст"]))
    r = g.complete("резюме", СООБЩЕНИЕ)
    assert r.text == "просто текст" and r.data is None
