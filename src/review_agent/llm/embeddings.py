"""Интерфейс эмбеддингов и заглушка для тестов.

Реализация для GigaChat: GigaChatClient.embed(). Любая другая (например, локальная
модель) подключается через тот же интерфейс, остальной код не меняется.
"""
from __future__ import annotations

import hashlib
import re
from typing import Protocol


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class GigaChatEmbedder:
    """Обёртка над GigaChatClient.embed с фиксированной моделью эмбеддингов."""

    def __init__(self, client, model: str = "Embeddings"):
        self._client = client
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed(texts, model=self._model)


class FakeEmbedder:
    """Заглушка для тестов: мешок слов в 64-мерном пространстве (хеш слова → координата).

    Похожие по словам тексты получают близкие векторы, поэтому кластеризацию можно
    проверять без сети.
    """

    def __init__(self, размерность: int = 64):
        self._n = размерность
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        результат = []
        for t in texts:
            v = [0.0] * self._n
            for слово in re.findall(r"\w+", t.lower()):
                v[int(hashlib.md5(слово.encode()).hexdigest(), 16) % self._n] += 1.0
            результат.append(v)
        return результат
