"""Настройки: какие модели доступны и как к ним подключаться.

Ключи берутся из переменных окружения или из файла .env в корне проекта (NFR-7).
"""
import os
from pathlib import Path

from review_agent.llm.base import LLMClient

ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
MODEL_NAMES = ("gigachat", "yandex", "qwen")


def load_env(path: Path | None = None) -> None:
    """Читает файл .env (строки ВИДА=значение). Уже заданные переменные не перезаписывает."""
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def _need(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"Не задана переменная окружения {name}. См. файл .env.example.")
    return value


def _price(prefix: str) -> dict:
    return {
        "price_in": float(os.environ.get(f"{prefix}_PRICE_IN", 0) or 0),
        "price_out": float(os.environ.get(f"{prefix}_PRICE_OUT", 0) or 0),
    }


def make_client(name: str) -> LLMClient:
    """Создаёт клиента модели по короткому имени: gigachat, yandex или qwen."""
    load_env()
    if name == "gigachat":
        from review_agent.llm.gigachat import GigaChatClient

        return GigaChatClient(
            auth_key=_need("GIGACHAT_AUTH_KEY"),
            model=os.environ.get("GIGACHAT_MODEL", "GigaChat-2-Pro"),
            scope=os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
            verify=os.environ.get("GIGACHAT_CA_BUNDLE") or True,
            **_price("GIGACHAT"),
        )
    from review_agent.llm.openai_compat import OpenAICompatClient

    if name == "yandex":
        folder = _need("YANDEX_FOLDER_ID")
        model = os.environ.get("YANDEX_MODEL", "yandexgpt/latest")
        return OpenAICompatClient(
            name="yandex",
            base_url="https://llm.api.cloud.yandex.net/v1",
            model=f"gpt://{folder}/{model}",
            api_key=_need("YANDEX_API_KEY"),
            **_price("YANDEX"),
        )
    if name == "qwen":
        return OpenAICompatClient(
            name="qwen",
            base_url=os.environ.get("QWEN_BASE_URL", "http://localhost:11434/v1"),
            model=_need("QWEN_MODEL"),
            api_key=os.environ.get("QWEN_API_KEY", ""),
            **_price("QWEN"),
        )
    raise ValueError(f"Неизвестная модель «{name}». Доступные: {', '.join(MODEL_NAMES)}.")
