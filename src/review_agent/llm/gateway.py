"""LLM-шлюз: проверка ответа по схеме, повторы, кэш, журнал, подсчёт стоимости (NFR-3…NFR-6)."""
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .base import LLMClient, LLMError

T = TypeVar("T", bound=BaseModel)


class FormatError(Exception):
    """Модель так и не вернула ответ в нужном формате за отведённое число попыток."""


@dataclass
class Usage:
    """Накопленная статистика шлюза за запуск."""

    calls: int = 0
    cached: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    seconds: float = 0.0
    format_errors: int = 0  # ответы, не прошедшие проверку формата (включая исправленные повтором)
    failed: int = 0  # запросы, которые не удалось выполнить совсем
    by_step: dict = field(default_factory=dict)


def extract_json(text: str) -> str:
    """Вырезает JSON из ответа: модели часто оборачивают его в ```json … ``` или добавляют пояснения."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("в ответе нет JSON-объекта")
    return text[start : end + 1]


class LLMGateway:
    def __init__(
        self,
        client: LLMClient,
        journal_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        format_retries: int = 2,
        network_retries: int = 3,
        retry_pause: float = 2.0,
        temperature: float = 0.0,
    ):
        self.client = client
        self.usage = Usage()
        self._journal = Path(journal_path) if journal_path else None
        self._cache = Path(cache_dir) if cache_dir else None
        self._format_retries = format_retries
        self._network_retries = network_retries
        self._retry_pause = retry_pause
        self._temperature = temperature
        for folder in (self._journal.parent if self._journal else None, self._cache):
            if folder:
                folder.mkdir(parents=True, exist_ok=True)

    # --- кэш: одинаковый запрос к той же модели второй раз не оплачивается ---
    def _cache_file(self, messages: list[dict]) -> Path | None:
        if not self._cache:
            return None
        raw = json.dumps([self.client.model, self._temperature, messages], ensure_ascii=False, sort_keys=True)
        return self._cache / (hashlib.sha256(raw.encode("utf-8")).hexdigest() + ".json")

    def _log(self, record: dict) -> None:
        if self._journal:
            with self._journal.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _call(self, step: str, messages: list[dict], max_tokens: int) -> str:
        """Один запрос к модели с повторами при сетевых ошибках."""
        cache_file = self._cache_file(messages)
        if cache_file and cache_file.exists():
            self.usage.cached += 1
            return json.loads(cache_file.read_text(encoding="utf-8"))["text"]

        last_error = None
        for attempt in range(1, self._network_retries + 1):
            started = time.perf_counter()
            try:
                reply = self.client.complete(messages, temperature=self._temperature, max_tokens=max_tokens)
            except LLMError as error:
                last_error = error
                self._log({"time": datetime.now().isoformat(timespec="seconds"), "step": step,
                           "model": self.client.model, "attempt": attempt, "error": str(error)})
                if attempt < self._network_retries:
                    time.sleep(self._retry_pause * attempt)
                continue
            seconds = time.perf_counter() - started
            cost = reply.tokens_in / 1000 * self.client.price_in + reply.tokens_out / 1000 * self.client.price_out
            usage = self.usage
            usage.calls += 1
            usage.tokens_in += reply.tokens_in
            usage.tokens_out += reply.tokens_out
            usage.cost += cost
            usage.seconds += seconds
            usage.by_step[step] = usage.by_step.get(step, 0) + 1
            self._log({"time": datetime.now().isoformat(timespec="seconds"), "step": step,
                       "model": self.client.model, "attempt": attempt, "tokens_in": reply.tokens_in,
                       "tokens_out": reply.tokens_out, "cost": round(cost, 4), "seconds": round(seconds, 2),
                       "messages": messages, "response": reply.text})
            if cache_file:
                cache_file.write_text(json.dumps({"text": reply.text}, ensure_ascii=False), encoding="utf-8")
            return reply.text
        self.usage.failed += 1
        raise last_error

    def ask_json(
        self,
        step: str,
        messages: list[dict],
        schema: type[T],
        check: Callable[[T], list[str]] | None = None,
        max_tokens: int = 2000,
    ) -> T:
        """Запрашивает у модели JSON и проверяет его.

        schema проверяет структуру ответа, check проверяет смысл (например, что цитата есть в тексте)
        и возвращает список замечаний. При ошибке запрос повторяется с текстом замечаний.
        """
        dialog = list(messages)
        problems: list[str] = []
        for _ in range(self._format_retries + 1):
            text = self._call(step, dialog, max_tokens)
            try:
                result = schema.model_validate_json(extract_json(text))
                problems = check(result) if check else []
            except ValidationError as error:
                problems = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors()[:5]]
            except ValueError as error:
                problems = [str(error)]
            if not problems:
                return result
            self.usage.format_errors += 1
            dialog = dialog + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": "Ответ не принят. Замечания:\n- " + "\n- ".join(problems[:10])
                 + "\nИсправь и верни только JSON в требуемом формате."},
            ]
        self.usage.failed += 1
        raise FormatError(f"Шаг «{step}»: ответ не прошёл проверку: {'; '.join(problems[:3])}")
