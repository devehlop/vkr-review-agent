"""LLM-шлюз: проверка ответа по схеме, повторы, кэш, журнал и стоимость.

Требования: NFR-3 (стоимость в журнале), NFR-5 (журнал всех запросов,
фиксированные параметры), NFR-6 (повтор при ошибке, проверка формата).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import defaultdict
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from review_agent.llm.base import LLMClient, LLMError, LLMResponse, Message

T = TypeVar("T", bound=BaseModel)
# Дополнительная проверка смысла ответа: возвращает список ошибок, пустой — всё хорошо
Validator = Callable[[T], list[str]]


class LLMFormatError(RuntimeError):
    """Модель так и не вернула ответ нужного формата."""

    def __init__(self, step: str, errors: list[str]):
        super().__init__(f"{step}: ответ не прошёл проверку: {'; '.join(errors)}")
        self.step = step
        self.errors = errors


class LLMCallRecord(BaseModel):
    """Одна запись журнала: один запрос к модели (таблица LLM_CALL)."""

    step: str
    provider: str
    model: str
    attempt: int
    prompt_hash: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    latency: float = 0.0
    cached: bool = False
    ok: bool = True
    error: str = ""
    response: str = ""


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> object:
    """Достаёт JSON из ответа модели: из блока ```json``` или по первой скобке."""
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        raise ValueError("в ответе нет JSON")
    start = min(starts)
    closing = "}" if text[start] == "{" else "]"
    end = text.rfind(closing)
    if end < start:
        raise ValueError("JSON не закрыт")
    return json.loads(text[start : end + 1])


def _hash(model: str, temperature: float, messages: list[Message]) -> str:
    payload = json.dumps(
        {"m": model, "t": temperature, "msgs": [(m.role, m.content) for m in messages]},
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class LLMGateway:
    def __init__(
        self,
        client: LLMClient,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_format_retries: int = 2,
        max_network_retries: int = 3,
        use_cache: bool = True,
        on_record: Callable[[LLMCallRecord], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_format_retries = max_format_retries
        self.max_network_retries = max_network_retries
        self.use_cache = use_cache
        self.on_record = on_record
        self._sleep = sleep
        self._cache: dict[str, LLMResponse] = {}
        self.records: list[LLMCallRecord] = []

    # --- журнал ---------------------------------------------------------

    def _log(self, record: LLMCallRecord) -> None:
        self.records.append(record)
        if self.on_record:
            self.on_record(record)

    def summary(self) -> dict[str, dict[str, float]]:
        """Итоги по шагам: число запросов, токены, стоимость, время, ошибки формата."""
        out: dict[str, dict[str, float]] = defaultdict(
            lambda: {"calls": 0, "tokens_in": 0, "tokens_out": 0, "cost": 0.0, "latency": 0.0, "errors": 0}
        )
        for r in self.records:
            s = out[r.step]
            s["calls"] += 0 if r.cached else 1
            s["tokens_in"] += r.tokens_in
            s["tokens_out"] += r.tokens_out
            s["cost"] += r.cost
            s["latency"] += r.latency
            s["errors"] += 0 if r.ok else 1
        return dict(out)

    # --- запросы --------------------------------------------------------

    def complete(self, step: str, messages: list[Message], attempt: int = 1) -> LLMResponse:
        """Один запрос с кэшем и повторами при сетевых ошибках."""
        key = _hash(self.client.model, self.temperature, messages)
        base = {"step": step, "provider": self.client.provider, "model": self.client.model, "attempt": attempt, "prompt_hash": key}
        if self.use_cache and key in self._cache:
            resp = self._cache[key]
            self._log(LLMCallRecord(**base, cached=True, response=resp.text))
            return resp

        delay = 2.0
        for net_try in range(1, self.max_network_retries + 1):
            started = time.perf_counter()
            try:
                resp = self.client.complete(messages, temperature=self.temperature, max_tokens=self.max_tokens)
            except LLMError as e:
                self._log(LLMCallRecord(**base, ok=False, error=f"сеть/API: {e}", latency=time.perf_counter() - started))
                if net_try == self.max_network_retries:
                    raise
                self._sleep(delay)
                delay *= 2
                continue
            latency = time.perf_counter() - started
            self._log(
                LLMCallRecord(
                    **base,
                    tokens_in=resp.tokens_in,
                    tokens_out=resp.tokens_out,
                    cost=self.client.cost(resp.tokens_in, resp.tokens_out),
                    latency=latency,
                    response=resp.text,
                )
            )
            if self.use_cache:
                self._cache[key] = resp
            return resp
        raise AssertionError("недостижимо")

    def structured(
        self,
        step: str,
        messages: list[Message],
        schema: type[T],
        validate: Validator | None = None,
    ) -> T:
        """Запрос с ответом в JSON по схеме pydantic.

        Если ответ не разбирается или не проходит проверку, модели
        возвращается текст ошибки и запрос повторяется (не более
        max_format_retries раз).
        """
        dialog = list(messages)
        errors: list[str] = []
        for attempt in range(1, self.max_format_retries + 2):
            resp = self.complete(step, dialog, attempt=attempt)
            try:
                parsed = schema.model_validate(extract_json(resp.text))
                errors = validate(parsed) if validate else []
            except (ValueError, ValidationError) as e:
                errors = [_short_error(e)]
            if not errors:
                return parsed
            # Отмечаем в журнале, что ответ не прошёл проверку
            self.records[-1].ok = False
            self.records[-1].error = "; ".join(errors)[:1000]
            dialog = dialog + [
                Message("assistant", resp.text),
                Message(
                    "user",
                    "Ответ не прошёл проверку:\n- "
                    + "\n- ".join(errors[:10])
                    + "\nИсправь ошибки и верни только JSON в том же формате.",
                ),
            ]
        raise LLMFormatError(step, errors)


def _short_error(e: Exception) -> str:
    if isinstance(e, ValidationError):
        parts = []
        for err in e.errors()[:5]:
            loc = ".".join(str(x) for x in err["loc"])
            parts.append(f"{loc}: {err['msg']}")
        return "неверная структура JSON: " + "; ".join(parts)
    return f"не удалось разобрать JSON: {e}"
