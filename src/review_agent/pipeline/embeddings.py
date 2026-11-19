"""Векторные представления текста для кластеризации проблем (шаг 6).

Два варианта с одинаковым интерфейсом:
- TfidfEmbedder — символьные n-граммы, без скачивания моделей. Годится для
  отладки и как базовая линия; плохо понимает синонимы («маломерит» и
  «мал на размер» для него разные).
- SentenceTransformerEmbedder — нейросетевая модель (по умолчанию небольшая
  русскоязычная rubert-tiny2). Нужна зависимость `pip install -e ".[embeddings]"`.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str
    # Порог косинусного расстояния для кластеризации по умолчанию: у разных
    # способов представления расстояния разные. Подбирается на тестировании.
    default_threshold: float

    def embed(self, texts: list[str]) -> np.ndarray:
        """Матрица (число текстов × размерность), строки нормированы по длине."""
        ...


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class TfidfEmbedder:
    name = "tfidf-char"
    # Символьные n-граммы дают большие расстояния даже между похожими фразами
    default_threshold = 0.8

    def __init__(self, ngram_range: tuple[int, int] = (2, 5)):
        self.ngram_range = ngram_range

    def embed(self, texts: list[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer

        if not texts:
            return np.zeros((0, 0))
        prepared = [t.lower().replace("ё", "е") for t in texts]
        # Словарь строится по самим текстам: для одной группы упоминаний этого достаточно
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=self.ngram_range, sublinear_tf=True)
        return _normalize(vec.fit_transform(prepared).toarray())


class SentenceTransformerEmbedder:
    # Замер 19.11.2026 на демо-наборе (docs/experiments.md): при 0,4 группы
    # осмысленные, при 0,5 и выше атрибут сливается в один кластер
    default_threshold = 0.4

    def __init__(self, model: str = "cointegrated/rubert-tiny2"):
        from sentence_transformers import SentenceTransformer

        self.name = model
        self._model = SentenceTransformer(model)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0))
        return _normalize(np.asarray(self._model.encode(texts, show_progress_bar=False)))
