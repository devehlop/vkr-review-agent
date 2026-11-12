"""Заглушка модели для сквозных тестов: отвечает по простым правилам вместо настоящей LLM.

Определяет шаг конвейера по тексту запроса и возвращает ответ в нужном формате.
Поведение настраивается: можно заставить критика отклонять рекомендации.
"""
import json
import re

RULES = [  # (подстрока в предложении, атрибут, тональность)
    ("расход", "Качество пошива", -1), ("разош", "Качество пошива", -1), ("ровные", "Качество пошива", 1),
    ("маломерит", "Размер и посадка", -1), ("в размер", "Размер и посадка", 1),
    ("приятная", "Ткань и материал", 1), ("мягкая", "Ткань и материал", 1),
]


class ScriptedModel:
    def __init__(self, critic_scores: list[int] | None = None, invent_quote: bool = False):
        self.critic_scores = critic_scores or [5]  # оценки критика по очереди, последняя повторяется
        self.invent_quote = invent_quote
        self.steps: list[str] = []

    def __call__(self, messages: list[dict]) -> str:
        prompt = messages[0]["content"]
        if "Составь справочник" in prompt:
            return self._say("taxonomy", {"attributes": [
                {"name": name, "description": "пояснение", "features": ["признак"]}
                for name in ("Качество пошива", "Размер и посадка", "Ткань и материал", "Доставка", "Общее впечатление")]})
        if "Для каждого предложения найди" in prompt:
            results = []
            for line in prompt.split("Предложения:\n")[1].strip().splitlines():
                item = json.loads(line)
                low = item["текст"].lower()
                found = [{"attribute": a, "feature": "", "sentiment": s, "quote": item["текст"]}
                         for key, a, s in RULES if key in low]
                results.append({"id": item["id"], "mentions": found})
            return self._say("extract", {"results": results})
        if "Сформулируй общую проблему" in prompt:
            attribute = re.search(r"характеристику товара: «(.+?)»", prompt).group(1)
            return self._say("issue_title", {"title": f"Жалобы: {attribute.lower()}"})
        if "Предложи продавцу" in prompt:
            first = re.search(r"^(\d+)\. (.+)$", prompt.split("Предложения из отзывов (номер и текст):\n")[1], re.M)
            quote = "такого в отзывах нет" if self.invent_quote else first.group(2)
            revised = "Предыдущая версия" in prompt
            return self._say("recommend", {"recommendations": [{
                "action": "Исправленное действие" if revised else "Усилить швы двойной строчкой",
                "type": "товар", "rationale": "Покупатели жалуются", "expected_effect": "Меньше возвратов",
                "effort": 2, "quotes": [{"sentence_id": int(first.group(1)), "quote": quote}]}]})
        if "Оцени рекомендацию" in prompt:
            index = min(self.steps.count("critic"), len(self.critic_scores) - 1)
            score = self.critic_scores[index]
            return self._say("critic", {"scores": dict.fromkeys(("specificity", "grounding", "feasibility", "novelty"), score),
                                        "comment": "" if score >= 4 else "слишком общо"})
        raise AssertionError("неизвестный запрос: " + prompt[:80])

    def _say(self, step: str, data: dict) -> str:
        self.steps.append(step)
        return json.dumps(data, ensure_ascii=False)
