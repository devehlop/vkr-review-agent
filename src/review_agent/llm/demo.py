"""Демо-заглушка: отвечает на все промпты конвейера по простым правилам.

Нужна, чтобы без ключей API прогнать конвейер целиком и показать
интерфейс. Это не анализ: аспекты ищутся словарным методом, проблемы
называются по первой цитате, рекомендации шаблонные. В отчёте такой
запуск помечается как демонстрационный.
"""

from __future__ import annotations

import json
import re

from review_agent.evaluation.baseline import ATTRIBUTE_STEMS, predict
from review_agent.llm.base import Message
from review_agent.llm.fake import FakeLLMClient
from review_agent.taxonomy import load_taxonomy

_TAX = load_taxonomy("clothing_v1")
_NAME = {a.code: a.attribute for a in _TAX.attributes}


def _section(text: str, start: str, end: str | None = None) -> str:
    part = text.split(start, 1)[-1]
    return part.split(end, 1)[0] if end else part


def _quotes(text: str) -> list[str]:
    return re.findall(r"^- (.+)$", text, flags=re.MULTILINE)


def demo_responder(messages: list[Message]) -> str:
    system = messages[0].content if messages[0].role == "system" else ""
    user = messages[-1].content
    # Повтор после ошибки: исходный запрос — первое сообщение пользователя
    first_user = next(m.content for m in messages if m.role == "user")

    if "Строишь справочник" in system:
        return json.dumps({"attributes": [{"attribute": a.attribute, "features": a.features} for a in _TAX.attributes]}, ensure_ascii=False)

    if "найти в каждом предложении" in system:
        block = _section(first_user, "Предложения:")
        sentences = dict(re.findall(r"^(\d+)\. (.+)$", block, flags=re.MULTILINE))
        labels = predict(sentences)
        results = []
        for sid, text in sentences.items():
            mentions = [{"attribute": _NAME[code], "feature": "другое", "sentiment": s, "quote": text}
                        for code, s in labels[sid].items()]
            results.append({"id": sid, "mentions": mentions})
        return json.dumps({"results": results}, ensure_ascii=False)

    if "Формулируешь проблемы" in system:
        q = _quotes(_section(first_user, "говорят примерно об одном.", "Сформулируй"))[0]
        words = re.findall(r"\w+", q)[:8]
        if len(words) < 2:
            words = ["Жалобы:", *words, "и похожее"]
        return json.dumps({"title": " ".join(words)}, ensure_ascii=False)

    if "строгий рецензент" in system:
        return json.dumps({"scores": {"specificity": 4, "grounding": 4, "feasibility": 4, "novelty": 4},
                           "comment": "демо-оценка"}, ensure_ascii=False)

    if "Предлагаешь конкретные" in system:
        title = re.search(r"Проблема товара: «(.+?)»", first_user).group(1)
        quotes = _quotes(_section(first_user, "Цитаты покупателей:", "Предложи"))[:2]
        rec = {"action": f"Разобраться с жалобами «{title}» и описать изменения в карточке товара",
               "type": "товар", "rationale": "Демо-обоснование по цитатам покупателей",
               "expected_effect": "меньше жалоб", "effort": 2, "quotes": quotes}
        return json.dumps({"recommendations": [rec]}, ensure_ascii=False)

    if "сильные стороны" in system:
        attr = re.search(r"хвалят атрибут «(.+?)»", first_user).group(1)
        quotes = _quotes(_section(first_user, "Цитаты:", "Предложи"))[:2]
        rec = {"action": f"Подчеркнуть в описании и на фото: {attr.lower()}", "type": "карточка товара",
               "rationale": "Демо-обоснование по цитатам покупателей", "expected_effect": "выше конверсия",
               "effort": 1, "quotes": quotes}
        return json.dumps({"recommendations": [rec]}, ensure_ascii=False)

    raise AssertionError(f"Демо-заглушка не знает такого промпта: {system[:80]!r}")


def demo_client() -> FakeLLMClient:
    return FakeLLMClient(demo_responder, model="demo")


__all__ = ["ATTRIBUTE_STEMS", "demo_client", "demo_responder"]
