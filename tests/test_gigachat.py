"""Тесты клиента GigaChat без сети: HTTP подменён заглушкой."""
import time

import pytest

from review_agent.llm.base import FakeLLMClient, Message
from review_agent.llm.gigachat import GigaChatClient, GigaChatError


class ФейковыйОтвет:
    def __init__(self, код, данные=None):
        self.status_code = код
        self._данные = данные or {}

    def json(self):
        return self._данные


class ФейковаяСессия:
    def __init__(self, ответы):
        self.ответы = list(ответы)
        self.вызовы = []

    def post(self, url, **kwargs):
        self.вызовы.append((url, kwargs))
        return self.ответы.pop(0)


def токен(секунд=1800):
    return ФейковыйОтвет(200, {"access_token": "T", "expires_at": (time.time() + секунд) * 1000})


def ответ_модели():
    return ФейковыйОтвет(200, {
        "model": "GigaChat-2-Pro",
        "choices": [{"message": {"content": "привет"}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1},
    })


def test_токен_запрашивается_один_раз():
    сессия = ФейковаяСессия([токен(), ответ_модели(), ответ_модели()])
    клиент = GigaChatClient("ключ", session=сессия)
    клиент.chat([Message("user", "привет")])
    r = клиент.chat([Message("user", "ещё")])
    assert r.text == "привет" and r.prompt_tokens == 5
    assert sum("oauth" in url for url, _ in сессия.вызовы) == 1


def test_просроченный_токен_обновляется():
    сессия = ФейковаяСессия([токен(10), ответ_модели(), токен(), ответ_модели()])
    клиент = GigaChatClient("ключ", session=сессия)
    клиент.chat([Message("user", "а")])
    клиент.chat([Message("user", "б")])
    assert sum("oauth" in url for url, _ in сессия.вызовы) == 2


def test_ошибка_не_содержит_ключа():
    сессия = ФейковаяСессия([ФейковыйОтвет(401)])
    клиент = GigaChatClient("секретный-ключ", session=сессия)
    with pytest.raises(GigaChatError) as e:
        клиент.chat([Message("user", "а")])
    assert "секретный-ключ" not in str(e.value)


def test_без_ключа_ошибка(monkeypatch):
    monkeypatch.delenv("GIGACHAT_CREDENTIALS", raising=False)
    with pytest.raises(GigaChatError):
        GigaChatClient()


def test_заглушка_отдаёт_ответы_по_очереди():
    f = FakeLLMClient(["один", "два"])
    assert f.chat([Message("user", "x")]).text == "один"
    assert f.chat([Message("user", "y")]).text == "два"


def test_expires_at_в_секундах_тоже_работает():
    # Пример из документации Сбера: expires_at в секундах.
    в_секундах = ФейковыйОтвет(200, {"access_token": "T", "expires_at": time.time() + 1800})
    сессия = ФейковаяСессия([в_секундах, ответ_модели(), ответ_модели()])
    клиент = GigaChatClient("ключ", session=сессия)
    клиент.chat([Message("user", "а")])
    клиент.chat([Message("user", "б")])
    assert sum("oauth" in url for url, _ in сессия.вызовы) == 1


def test_строгий_режим_добавляет_response_format_только_если_включён():
    схема = {"type": "object"}
    выкл = ФейковаяСессия([токен(), ответ_модели()])
    GigaChatClient("ключ", session=выкл).chat([Message("user", "а")], json_schema=схема)
    assert "response_format" not in выкл.вызовы[-1][1]["json"]
    вкл = ФейковаяСессия([токен(), ответ_модели()])
    GigaChatClient("ключ", session=вкл, structured_output=True).chat([Message("user", "а")], json_schema=схема)
    assert вкл.вызовы[-1][1]["json"]["response_format"] == {"type": "json_schema", "schema": схема, "strict": True}


def test_finish_reason_читается():
    сессия = ФейковаяСессия([токен(), ответ_модели()])
    assert GigaChatClient("ключ", session=сессия).chat([Message("user", "а")]).finish_reason == "length"
