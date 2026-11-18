"""Шаг 7 конвейера, текстовая часть: короткое резюме сравнения с конкурентами (FR-12)."""
import re

from pydantic import BaseModel

from review_agent.config import load_prompt
from review_agent.llm.base import LLMError
from review_agent.llm.gateway import FormatError, LLMGateway
from review_agent.pipeline.compare import NO_DATA, Comparison

PROMPT_VERSION = "compare_summary_v1"


class Summary(BaseModel):
    summary: str


def _number(share: float) -> int:
    return round(share * 100)


def summarize_comparison(gateway: LLMGateway, comparisons: list[Comparison], own_product: str,
                         category: str = "одежда", max_chars: int = 700) -> str | None:
    """Просит модель описать различия словами. Возвращает None, если сравнивать нечего или модель не справилась.

    Числа в резюме проверяются: каждый процент должен быть в таблице сравнения.
    Так модель не может приписать товару показатель, которого нет в данных.
    """
    rows = [c for c in comparisons if c.verdict != NO_DATA]
    if not rows:
        return None
    allowed = set()
    lines = []
    for row in rows:
        allowed.add(_number(row.own_neg_share))
        parts = [f"наш товар {_number(row.own_neg_share)} %"]
        for name, (neg, _) in row.competitors.items():
            allowed.add(_number(neg))
            parts.append(f"{name} {_number(neg)} %")
        lines.append(f"- {row.attribute}: {', '.join(parts)}. Наш товар {row.verdict}.")

    def check(result: Summary) -> list[str]:
        text = result.summary.strip()
        problems = []
        if not text:
            problems.append("пустое резюме")
        if len(text) > max_chars:
            problems.append(f"резюме длиннее {max_chars} знаков")
        extra = sorted({int(n) for n in re.findall(r"(\d+)\s*%", text)} - allowed)
        if extra:
            problems.append(f"чисел {extra} нет в таблице: используй только числа из таблицы")
        return problems

    prompt = load_prompt(PROMPT_VERSION).format(category=category, product=own_product, table="\n".join(lines))
    try:
        return gateway.ask_json("compare_summary", [{"role": "user", "content": prompt}], Summary,
                                check=check, max_tokens=500).summary.strip()
    except (FormatError, LLMError):
        return None
