"""LLM-шлюз: единая точка вызова модели.

Что делает (NFR-3…NFR-6 из docs/tz.md):
- повторяет запрос при ошибке сети/сервиса и при невалидном ответе;
- проверяет JSON-ответ по схеме (jsonschema);
- кэширует ответы в SQLite (одинаковый запрос второй раз не тратит токены);
- пишет журнал всех обращений (таблица llm_call): шаг, токены, стоимость, время, ошибка.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Callable

import jsonschema

from .base import LLMClient, Message


class LLMGatewayError(RuntimeError):
    """Все попытки исчерпаны. Вызывающий код может пропустить запись и идти дальше."""


@dataclass
class GatewayResult:
    text: str
    data: Any = None          # разобранный JSON, если передана схема
    from_cache: bool = False


_СХЕМА_БД = """
CREATE TABLE IF NOT EXISTS llm_cache (
    key TEXT PRIMARY KEY, text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS llm_call (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, step TEXT NOT NULL, model TEXT NOT NULL,
    tokens_in INTEGER NOT NULL, tokens_out INTEGER NOT NULL,
    cost REAL NOT NULL, latency REAL NOT NULL,
    from_cache INTEGER NOT NULL, ok INTEGER NOT NULL, error TEXT
);
"""


_ПАРА = {"}": "{", "]": "["}
_ЗАКРЫТИЕ = {"{": "}", "[": "]"}


def починить_скобки(text: str) -> str:
    """Чинит типичную ошибку LLM: пропущенные закрывающие скобки в JSON.

    Идёт по тексту со стеком открытых скобок (строки пропускаются). Если закрывающая
    скобка не подходит к верхней открытой, недостающие закрывающие вставляются перед ней;
    если текст кончился раньше, закрывающие дописываются в конце. Содержимое строк и
    значения не меняются. Результат всё равно проверяется по схеме.
    """
    очищенный = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    стек: list[str] = []
    результат: list[str] = []
    в_строке = False
    экран = False
    for символ in очищенный:
        if в_строке:
            результат.append(символ)
            if экран:
                экран = False
            elif символ == "\\":
                экран = True
            elif символ == '"':
                в_строке = False
            continue
        if символ == '"':
            в_строке = True
        elif символ in _ЗАКРЫТИЕ:
            стек.append(символ)
        elif символ in _ПАРА:
            if _ПАРА[символ] in стек:
                while стек and стек[-1] != _ПАРА[символ]:
                    результат.append(_ЗАКРЫТИЕ[стек.pop()])
                стек.pop()
            else:
                continue          # лишняя закрывающая скобка без пары: выбрасываем
        результат.append(символ)
    if в_строке:
        результат.append('"')
    результат.extend(_ЗАКРЫТИЕ[с] for с in reversed(стек))
    return "".join(результат)


def _разобрать_json(text: str) -> tuple[Any, bool]:
    """Достаёт JSON из ответа (модели часто оборачивают его в ```json ... ```).

    Возвращает (данные, были_ли_починены_скобки).
    """
    очищенный = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(очищенный), False
    except json.JSONDecodeError:
        return json.loads(починить_скобки(text)), True


class LLMGateway:
    def __init__(self, client: LLMClient, *, db_path: str = ":memory:",
                 max_retries: int = 3, backoff: float = 1.0,
                 price_per_1k_in: float = 0.0, price_per_1k_out: float = 0.0,
                 sleep: Callable[[float], None] = time.sleep):
        self._client = client
        self._max_retries = max_retries
        self._backoff = backoff
        self._price_in = price_per_1k_in
        self._price_out = price_per_1k_out
        self._sleep = sleep
        self._db = sqlite3.connect(db_path)
        self._db.executescript(_СХЕМА_БД)
        # Журнал в файле копится между запусками; статистика по умолчанию считается
        # только по вызовам этого экземпляра шлюза (с id больше стартового).
        self._старт_id = self._db.execute("SELECT COALESCE(MAX(id), 0) FROM llm_call").fetchone()[0]

    # --- служебное -------------------------------------------------------
    def _модель(self) -> str:
        return getattr(self._client, "model", "unknown")

    def _ключ_кэша(self, messages: list[Message], temperature: float,
                   max_tokens: int, schema: dict | None) -> str:
        сырьё = json.dumps(
            [self._модель(), [(m.role, m.content) for m in messages],
             temperature, max_tokens, schema],
            ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(сырьё.encode("utf-8")).hexdigest()

    def _записать_вызов(self, step: str, tin: int, tout: int, latency: float,
                        from_cache: bool, ok: bool, error: str | None = None) -> None:
        стоимость = 0.0 if from_cache else (
            tin / 1000 * self._price_in + tout / 1000 * self._price_out)
        self._db.execute(
            "INSERT INTO llm_call (ts, step, model, tokens_in, tokens_out, cost,"
            " latency, from_cache, ok, error) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (time.time(), step, self._модель(), tin, tout, стоимость, latency,
             int(from_cache), int(ok), error))
        self._db.commit()

    # --- основной метод --------------------------------------------------
    def complete(self, step: str, messages: list[Message], *, schema: dict | None = None,
                 temperature: float = 0.0, max_tokens: int = 1024) -> GatewayResult:
        """Вызывает модель. С schema ответ должен быть JSON, подходящим под схему.

        Температура по умолчанию 0: нужна воспроизводимость (NFR-5).
        """
        ключ = self._ключ_кэша(messages, temperature, max_tokens, schema)
        строка = self._db.execute("SELECT text FROM llm_cache WHERE key=?", (ключ,)).fetchone()
        if строка:
            текст = строка[0]
            self._записать_вызов(step, 0, 0, 0.0, from_cache=True, ok=True)
            return GatewayResult(текст, _разобрать_json(текст)[0] if schema else None, True)

        последняя_ошибка = ""
        for попытка in range(self._max_retries):
            начало = time.time()
            диагностика = ""
            try:
                # При температуре 0 модель повторяет тот же ответ, поэтому после сбоя
                # на следующих попытках температура немного растёт.
                t = temperature if попытка == 0 else min(1.0, temperature + 0.3 * попытка)
                ответ = self._client.chat(messages, temperature=t, max_tokens=max_tokens,
                                          json_schema=schema)
                диагностика = (f"длина={len(ответ.text)}, finish_reason={ответ.finish_reason or '?'}, "
                               f"начало={ответ.text[:120]!r}, конец={ответ.text[-120:]!r}")
                данные, починен = None, False
                if schema is not None:
                    данные, починен = _разобрать_json(ответ.text)
                    jsonschema.validate(данные, schema)
            except Exception as e:  # сеть, HTTP-ошибка, битый JSON, нарушение схемы
                последняя_ошибка = f"{type(e).__name__}: {e}"[:200]
                if диагностика:   # для диагностики: что именно пришло и чем закончилось
                    последняя_ошибка += f" | {диагностика}"
                self._записать_вызов(step, 0, 0, time.time() - начало,
                                     from_cache=False, ok=False, error=последняя_ошибка)
                if попытка < self._max_retries - 1:
                    self._sleep(self._backoff * 2 ** попытка)
                continue
            self._db.execute("INSERT OR REPLACE INTO llm_cache VALUES (?, ?)", (ключ, ответ.text))
            self._записать_вызов(step, ответ.prompt_tokens, ответ.completion_tokens,
                                 time.time() - начало, from_cache=False, ok=True,
                                 error="починены скобки JSON" if починен else None)
            return GatewayResult(ответ.text, данные)

        raise LLMGatewayError(
            f"Шаг «{step}»: {self._max_retries} попыток неудачны. Последняя ошибка: {последняя_ошибка}")

    def ошибки(self, лимит: int = 5) -> list[tuple[str, str]]:
        """Последние ошибки текущего запуска: (шаг, текст). Нужны для диагностики."""
        строки = self._db.execute(
            "SELECT step, error FROM llm_call WHERE id > ? AND ok = 0 ORDER BY id DESC LIMIT ?",
            (self._старт_id, лимит)).fetchall()
        return [(ш, e) for ш, e in строки]

    # --- статистика ------------------------------------------------------
    def stats(self, *, весь_журнал: bool = False) -> dict:
        """Итоги: число вызовов, попаданий в кэш, ошибок, токены, стоимость.

        По умолчанию только за текущий запуск; весь_журнал=True считает все запуски.
        """
        с_id = 0 if весь_журнал else self._старт_id
        в = self._db.execute(
            "SELECT COUNT(*), COALESCE(SUM(from_cache),0), COALESCE(SUM(1-ok),0),"
            " COALESCE(SUM(tokens_in),0), COALESCE(SUM(tokens_out),0), COALESCE(SUM(cost),0),"
            " COALESCE(SUM(ok = 1 AND error LIKE 'починены%'), 0)"
            " FROM llm_call WHERE id > ?", (с_id,)).fetchone()
        return {"calls": в[0], "cache_hits": в[1], "errors": в[2],
                "tokens_in": в[3], "tokens_out": в[4], "cost": в[5], "repaired": в[6]}
