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

    def chat(self, messages, *, temperature=0.2, max_tokens=1024, json_schema=None):
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


def test_статистика_только_за_текущий_запуск(tmp_path):
    путь = str(tmp_path / "журнал.sqlite")
    первый = LLMGateway(Сбойный(падений=0), db_path=путь, sleep=lambda _: None)
    первый.complete("а", СООБЩЕНИЕ, temperature=0.1)
    второй = LLMGateway(Сбойный(падений=0), db_path=путь, sleep=lambda _: None)
    второй.complete("а", СООБЩЕНИЕ, temperature=0.2)   # другие параметры: кэш не сработает
    assert второй.stats()["calls"] == 1 and второй.stats()["tokens_in"] == 100
    assert второй.stats(весь_журнал=True)["calls"] == 2


def test_ошибки_содержат_начало_ответа_для_диагностики():
    g = шлюз(FakeLLMClient(["это не json, а просто слова"] * 2), max_retries=2)
    with pytest.raises(LLMGatewayError):
        g.complete("шаг", СООБЩЕНИЕ, schema=СХЕМА)
    шаг, текст = g.ошибки()[0]
    assert шаг == "шаг" and "это не json" in текст and "JSONDecodeError" in текст
    assert "длина=" in текст and "конец=" in текст


def test_температура_растёт_на_повторах_после_невалидного_ответа():
    клиент = FakeLLMClient(["не json", "не json", '{"аспект": "цвет"}'])
    g = шлюз(клиент, max_retries=3)
    g.complete("а", СООБЩЕНИЕ, schema=СХЕМА, temperature=0.0)
    assert клиент.temperatures == [0.0, pytest.approx(0.3), pytest.approx(0.6)]
