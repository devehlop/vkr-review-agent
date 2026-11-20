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


class LocalEmbedder:
    """Локальная модель эмбеддингов (sentence-transformers): без ключей, лимитов и сети.

    По умолчанию intfloat/multilingual-e5-small: многоязычная, понимает русский,
    вектор 384, модель около 0,5 ГБ (скачивается один раз при первом запуске).
    Моделям семейства e5 нужна приставка «query: » перед текстом.
    Пакеты sentence-transformers и torch импортируются только при первом использовании.
    """

    def __init__(self, модель: str = "intfloat/multilingual-e5-small", приставка: str = "query: ",
                 загрузчик=None):
        self._имя = модель
        self._приставка = приставка
        self._загрузчик = загрузчик    # для тестов: функция, возвращающая объект с encode()
        self._модель = None

    def _получить_модель(self):
        if self._модель is None:
            if self._загрузчик is not None:
                self._модель = self._загрузчик(self._имя)
            else:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as e:
                    raise RuntimeError(
                        "Для локальных эмбеддингов установите: pip install sentence-transformers") from e
                self._модель = SentenceTransformer(self._имя)
        return self._модель

    def embed(self, texts: list[str]) -> list[list[float]]:
        векторы = self._получить_модель().encode(
            [self._приставка + t for t in texts], normalize_embeddings=True, show_progress_bar=False)
        return [list(map(float, v)) for v in векторы]


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
