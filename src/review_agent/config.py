"""Настройки системы.

Ключи API берутся только из переменных окружения (NFR-7), в коде и файлах их нет.
Пороговые значения подбираются на этапе тестирования, здесь стартовые значения из
docs/architecture.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Thresholds:
    # Минимальный размер кластера, чтобы он стал отдельной проблемой (шаг 6)
    min_issue_size: int = 3
    # Минимум реальных предложений в опоре проблемы (шаг 11)
    min_support: int = 3
    # Коэффициент приоритета, если у конкурентов проблема заметно реже (шаг 10)
    competitor_k: float = 1.5
    # «Заметно реже»: доля у конкурента не больше этой доли от своей
    competitor_ratio: float = 0.5
    # Сколько раз переспрашивать модель при ответе не по схеме (NFR-6)
    max_format_retries: int = 2
    # Сколько раз критик может вернуть рекомендацию на доработку (шаг 9)
    max_critic_rounds: int = 2


@dataclass(frozen=True)
class Settings:
    # Имя провайдера LLM: fake, gigachat, yandex, ollama (NFR-4)
    llm_provider: str = "fake"
    llm_model: str = ""
    temperature: float = 0.0
    # Размер пакета предложений для извлечения аспектов (шаг 4)
    extract_batch_size: int = 25
    db_path: str = "runs/review_agent.db"
    thresholds: Thresholds = field(default_factory=Thresholds)

    @classmethod
    def from_env(cls) -> "Settings":
        """Читает настройки из переменных окружения с префиксом REVIEW_AGENT_."""
        return cls(
            llm_provider=os.getenv("REVIEW_AGENT_LLM_PROVIDER", "fake"),
            llm_model=os.getenv("REVIEW_AGENT_LLM_MODEL", ""),
            temperature=float(os.getenv("REVIEW_AGENT_TEMPERATURE", "0")),
            extract_batch_size=int(os.getenv("REVIEW_AGENT_BATCH_SIZE", "25")),
            db_path=os.getenv("REVIEW_AGENT_DB", "runs/review_agent.db"),
        )
