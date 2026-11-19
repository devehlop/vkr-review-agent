"""Промпты в отдельных файлах с номером версии (NFR-5).

Файл <имя>.<версия>.md делится на части заголовками «## system» и «## user».
Подстановки пишутся как $имя (string.Template), чтобы фигурные скобки
в примерах JSON не мешали.
"""

from __future__ import annotations

import re
from importlib import resources
from string import Template

from review_agent.llm.base import Message


def load_prompt(name: str, version: str = "v1") -> dict[str, str]:
    text = resources.files(__package__).joinpath(f"{name}.{version}.md").read_text(encoding="utf-8")
    parts = re.split(r"^## (system|user)\s*$", text, flags=re.MULTILINE)
    # parts: [преамбула, "system", текст, "user", текст]
    sections = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
    if "user" not in sections:
        raise ValueError(f"В промпте {name}.{version} нет раздела «## user»")
    return sections


def render_prompt(name: str, version: str = "v1", **values: str) -> list[Message]:
    """Собирает сообщения для модели, подставляя значения. Пропущенная подстановка — ошибка."""
    sections = load_prompt(name, version)
    messages = []
    if "system" in sections:
        messages.append(Message("system", Template(sections["system"]).substitute(values)))
    messages.append(Message("user", Template(sections["user"]).substitute(values)))
    return messages
