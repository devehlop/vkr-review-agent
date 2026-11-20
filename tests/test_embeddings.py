"""Тесты локального эмбеддера на подменённой модели (скачивание модели не требуется)."""
import numpy as np
import pytest

from review_agent.llm.embeddings import LocalEmbedder


class ФейковаяМодель:
    def __init__(self):
        self.тексты = None

    def encode(self, тексты, normalize_embeddings, show_progress_bar):
        self.тексты = тексты
        assert normalize_embeddings is True
        return np.array([[1.0, 0.0] for _ in тексты], dtype=np.float32)


def test_приставка_добавляется_и_вектора_в_списках():
    модель = ФейковаяМодель()
    э = LocalEmbedder(загрузчик=lambda имя: модель)
    assert э.embed(["молния заедает", "цвет"]) == [[1.0, 0.0], [1.0, 0.0]]
    assert модель.тексты == ["query: молния заедает", "query: цвет"]


def test_модель_загружается_один_раз():
    загрузок = []
    э = LocalEmbedder(загрузчик=lambda имя: загрузок.append(имя) or ФейковаяМодель())
    э.embed(["а"]); э.embed(["б"])
    assert загрузок == ["intfloat/multilingual-e5-small"]


def test_без_пакета_понятная_подсказка(monkeypatch):
    import builtins
    настоящий = builtins.__import__

    def запрет(имя, *a, **k):
        if имя == "sentence_transformers":
            raise ImportError
        return настоящий(имя, *a, **k)

    monkeypatch.setattr(builtins, "__import__", запрет)
    with pytest.raises(RuntimeError, match="pip install sentence-transformers"):
        LocalEmbedder().embed(["а"])
