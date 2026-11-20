"""Агломеративная кластеризация по косинусному расстоянию (шаг 6).

Число кластеров заранее не задаётся: кластеры объединяются, пока расстояние между
ними не превысит порог. Связь average (среднее расстояние между всеми парами).
Реализация на numpy, рассчитана на десятки и сотни точек (упоминаний одного атрибута).
"""
from __future__ import annotations

import numpy as np


def кластеризовать(векторы: list[list[float]], порог: float = 0.35) -> list[list[int]]:
    """Возвращает кластеры как списки индексов, крупные первыми."""
    n = len(векторы)
    if n == 0:
        return []
    X = np.asarray(векторы, dtype=float)
    нормы = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.where(нормы == 0, 1.0, нормы)
    D = 1.0 - X @ X.T                      # косинусное расстояние
    np.fill_diagonal(D, np.inf)

    кластеры: list[list[int]] = [[i] for i in range(n)]
    дист = D.copy()
    живые = list(range(n))                 # индексы строк матрицы активных кластеров
    while len(живые) > 1:
        под = дист[np.ix_(живые, живые)]
        i, j = np.unravel_index(np.argmin(под), под.shape)
        if под[i, j] > порог:
            break
        a, b = живые[i], живые[j]
        na, nb = len(кластеры[a]), len(кластеры[b])
        # среднее расстояние (UPGMA): взвешенное по размерам объединяемых кластеров
        for k in живые:
            if k in (a, b):
                continue
            новое = (дист[a, k] * na + дист[b, k] * nb) / (na + nb)
            дист[a, k] = дист[k, a] = новое
        кластеры[a] = кластеры[a] + кластеры[b]
        кластеры[b] = []
        живые.remove(b)
    return sorted((кластеры[k] for k in живые), key=lambda c: (-len(c), c[0]))


def ближайшие_к_центру(векторы: list[list[float]], индексы: list[int], сколько: int) -> list[int]:
    """Индексы `сколько` точек кластера, ближайших к его центру (по косинусу)."""
    X = np.asarray([векторы[i] for i in индексы], dtype=float)
    X = X / np.where(np.linalg.norm(X, axis=1, keepdims=True) == 0, 1.0,
                     np.linalg.norm(X, axis=1, keepdims=True))
    центр = X.mean(axis=0)
    норма = np.linalg.norm(центр)
    центр = центр / норма if норма else центр
    порядок = np.argsort(-(X @ центр), kind="stable")
    return [индексы[i] for i in порядок[:сколько]]
