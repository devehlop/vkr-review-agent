"""Заглушка модели для сквозных тестов: отвечает по простым правилам вместо GigaChat.

В отличие от FakeLLMClient, не требует заранее знать порядок вызовов: шаг конвейера
определяется по тексту промпта. Используется в тестах полного прогона и для проверки
интерфейса без ключа.
"""
from __future__ import annotations

import json
import re

from review_agent.llm.base import LLMResponse

# (подстрока в предложении, атрибут, тональность)
ПРАВИЛА = [
    ("молни", "Фурнитура", "negative"), ("заеда", "Фурнитура", "negative"),
    ("маломер", "Размер и посадка", "negative"), ("в размер", "Размер и посадка", "positive"),
    ("коробк", "Упаковка", "negative"),
    ("тёпл", "Материал", "positive"), ("тепл", "Материал", "positive"),
]


class ScriptedLLMClient:
    def __init__(self, оценка_критика: int = 5):
        self.оценка_критика = оценка_критика
        self.шаги: list[str] = []

    def chat(self, messages, *, temperature=0.2, max_tokens=1024, json_schema=None) -> LLMResponse:
        промпт = messages[-1].content if hasattr(messages[-1], "content") else messages[-1]["content"]
        return LLMResponse(text=json.dumps(self._ответ(промпт), ensure_ascii=False), model="scripted")

    def _ответ(self, п: str) -> dict:
        if "двухуровневый справочник" in п:
            self.шаги.append("taxonomy")
            return {"attributes": [{"name": и, "features": ["признак"]} for и in
                                   ("Фурнитура", "Размер и посадка", "Упаковка", "Материал")]}
        if "Предложения:\n" in п and '"items"' in п:
            self.шаги.append("extract")
            элементы = []
            for строка in п.split("Предложения:\n", 1)[1].strip().splitlines():
                номер, текст = строка.split(". ", 1)
                низ = текст.lower()
                элементы.append({"i": int(номер), "mentions": [
                    {"attribute": а, "sentiment": т} for ключ, а, т in ПРАВИЛА if ключ in низ]})
            return {"items": элементы}
        if "Тема претензий:" in п:
            self.шаги.append("issue_title")
            тема = re.search(r"Тема претензий: (.+?)\.\n", п).group(1)
            return {"title": f"Жалобы: {тема.lower()}"}
        if "Предложи продавцу" in п:
            self.шаги.append("recommend")
            первая = re.search(r"^1\. (.+)$", п, re.M).group(1)
            return {"recommendations": [{
                "action": "Проверить партию и заменить слабую деталь у поставщика", "type": "товар",
                "rationale": "Жалобы повторяются", "expected_effect": "Меньше возвратов", "effort": 2,
                "quotes": [{"n": 1, "text": первая[:25]}]}]}
        if "Оцени рекомендацию" in п:
            self.шаги.append("critic")
            о = self.оценка_критика
            return {"scores": dict.fromkeys(("specificity", "grounding", "feasibility", "novelty"), о),
                    "comment": "" if о >= 4 else "слишком общо"}
        raise AssertionError("Неизвестный запрос: " + п[:120])
