"""Вспомогательные функции веб-интерфейса (app/streamlit_app.py).

Вынесены из файла интерфейса, чтобы их можно было проверить тестами
и подменить модель заглушкой при проверке интерфейса без ключа.
"""
from __future__ import annotations

import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

КОРЕНЬ = Path(__file__).resolve().parents[2]
ПАПКА_ПРОГОНОВ = КОРЕНЬ / "runs"          # результаты интерфейса; в .gitignore: там тексты отзывов
ИМЕНА_РЕЗУЛЬТАТОВ = ("issues.json", "aggregate.csv")


def ключ_задан() -> bool:
    return bool(os.environ.get("GIGACHAT_CREDENTIALS"))


def создать_клиента(модель: str = "GigaChat-2-Pro"):
    """Клиент GigaChat. Ключ берётся из переменной окружения GIGACHAT_CREDENTIALS."""
    from review_agent.llm.gigachat import GigaChatClient
    return GigaChatClient(model=модель)


def создать_эмбеддер(вид: str, клиент=None):
    """local — модель на компьютере (бесплатно, нужен sentence-transformers); gigachat — платный пакет."""
    from review_agent.llm.embeddings import GigaChatEmbedder, LocalEmbedder
    return GigaChatEmbedder(клиент) if вид == "gigachat" else LocalEmbedder()


def сохранить_загруженный(данные: bytes, имя: str) -> Path:
    """Загруженный в браузере файл кладётся во временную папку: загрузчик читает файлы с диска."""
    папка = Path(tempfile.mkdtemp(prefix="review_agent_"))
    путь = папка / Path(имя).name
    путь.write_bytes(данные)
    return путь


def папка_нового_прогона(товар: str, сейчас: datetime | None = None) -> Path:
    метка = (сейчас or datetime.now()).strftime("%Y-%m-%d_%H-%M")
    безопасное = re.sub(r"[^\w-]+", "_", товар, flags=re.UNICODE).strip("_")[:40] or "товар"
    return ПАПКА_ПРОГОНОВ / f"{метка}_{безопасное}"


def папки_с_результатами(*корни: Path) -> list[Path]:
    """Папки, где лежат issues.json и aggregate.csv: сохранённые прогоны и data/. Сначала новые."""
    найденные = []
    for корень in корни or (ПАПКА_ПРОГОНОВ, КОРЕНЬ / "data"):
        if not корень.is_dir():
            continue
        for папка in [корень, *sorted(п for п in корень.iterdir() if п.is_dir())]:
            if all((папка / имя).is_file() for имя in ИМЕНА_РЕЗУЛЬТАТОВ):
                найденные.append(папка)
    return sorted(set(найденные), key=lambda п: п.stat().st_mtime, reverse=True)
