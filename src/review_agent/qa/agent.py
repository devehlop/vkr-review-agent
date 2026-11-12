"""Агент вопросов и ответов (FR-21).

Работает по схеме «решение модели → вызов инструмента → наблюдение»: модель сама выбирает,
какие данные запросить, и повторяет это, пока не будет готова ответить.
Ответ должен ссылаться на предложения, которые вернули инструменты.
"""
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel

from review_agent.config import load_prompt
from review_agent.llm.base import LLMError
from review_agent.llm.gateway import FormatError, LLMGateway
from review_agent.pipeline.compare import NO_DATA
from review_agent.pipeline.orchestrator import PipelineResult
from review_agent.pipeline.recommend import percent

PROMPT_VERSION = "qa_v1"
MAX_STEPS = 5
TOOLS = ("search_reviews", "aspect_stats", "compare", "list_issues")


class Step(BaseModel):
    """Ответ модели на одном шаге: либо вызов инструмента, либо окончательный ответ."""

    tool: str | None = None
    args: dict[str, Any] = {}
    answer: str | None = None
    sentence_ids: list[int] = []


@dataclass
class Answer:
    text: str
    sentence_ids: list[int] = field(default_factory=list)
    tool_calls: list[tuple[str, dict]] = field(default_factory=list)  # что агент запрашивал: видно в интерфейсе
    ok: bool = True


STOP_WORDS = frozenset("""что как про для или при это эта эти есть был была были очень ещё еще уже там тут
    пишут пишет говорят говорит думают покупатели покупатель отзывы отзывах отзыв товар товара товаре
    почему какие какой какая каких сколько много мало чаще всего""".split())


def stems(text: str) -> set[str]:
    """Основы слов для поиска: первые пять букв слова. Грубая замена морфологии без словарей."""
    words = re.findall(r"[а-яa-z0-9]+", text.lower().replace("ё", "е"))
    return {w[:5] for w in words if len(w) >= 3 and w not in STOP_WORDS}


def lexical_scores(query: str, texts: list[str]) -> list[float]:
    """Поиск по словам: сумма весов слов запроса, найденных в тексте. Редкие слова весят больше.

    Находит только тексты с теми же словами. Запрос «размер» не найдёт «маломерит»:
    для поиска по смыслу нужны эмбеддинги модели или фильтр по характеристике.
    """
    text_stems = [stems(t) for t in texts]
    scores = []
    for found in text_stems:
        score = 0.0
        for stem in stems(query) & found:
            frequency = sum(stem in other for other in text_stems)
            score += math.log(1 + len(texts) / frequency)
        scores.append(score)
    return scores


class Toolbox:
    """Инструменты агента. Запоминает номера предложений, которые показал модели."""

    def __init__(self, result: PipelineResult, embed: Callable[[list[str]], list[list[float]]] | None = None):
        self.result = result
        self.embed = embed  # эмбеддинги модели для поиска по смыслу; без них поиск идёт по словам
        self.seen: set[int] = set()
        self._product = {r.id: r.product for r in result.reviews}

    def call(self, name: str, args: dict) -> str:
        try:
            return getattr(self, name)(**args)
        except TypeError as error:  # модель передала лишний или неверный аргумент
            return f"Ошибка вызова {name}: {error}"

    def search_reviews(self, query: str, attribute: str | None = None, sentiment: int | None = None,
                       product: str | None = None, limit: int = 8) -> str:
        product = product or self.result.own_product
        pool = []
        for sentence in self.result.sentences:
            if self._product.get(sentence.review_id) != product:
                continue
            found = self.result.mentions.get(sentence.id, [])
            if attribute and not any(m.attribute == attribute for m in found):
                continue
            if sentiment is not None and not any(
                    m.sentiment == int(sentiment) and (not attribute or m.attribute == attribute) for m in found):
                continue
            pool.append(sentence)
        if not pool:
            return "Ничего не найдено."
        order = self._rank(query, [s.text for s in pool])[:limit]
        if not order:  # по словам запроса ничего не нашлось, но фильтры заданы: показываем первые подходящие
            if not (attribute or sentiment is not None):
                return ("По словам запроса ничего не найдено. Попробуй другие слова "
                        "или укажи характеристику в аргументе attribute.")
            order = list(range(min(limit, len(pool))))
        chosen = [pool[i] for i in order]
        self.seen.update(s.id for s in chosen)
        return f"Найдено предложений: {len(pool)}, показаны {len(chosen)}:\n" + "\n".join(f"{s.id}. {s.text}" for s in chosen)

    def _rank(self, query: str, texts: list[str]) -> list[int]:
        """Индексы текстов по убыванию близости к запросу."""
        if self.embed:
            try:
                vectors = np.asarray(self.embed([query] + texts), dtype=float)
                norms = np.linalg.norm(vectors, axis=1)
                scores = vectors[1:] @ vectors[0] / np.where(norms[1:] * norms[0] == 0, 1, norms[1:] * norms[0])
                return [int(i) for i in np.argsort(-scores)]
            except LLMError:
                pass  # эмбеддинги недоступны: ищем по словам
        scores = lexical_scores(query, texts)
        return [i for i in sorted(range(len(texts)), key=lambda i: -scores[i]) if scores[i] > 0]

    def aspect_stats(self, attribute: str, product: str | None = None) -> str:
        product = product or self.result.own_product
        item = next((s for s in self.result.stats
                     if s.product == product and s.attribute == attribute and not s.feature), None)
        if item is None:
            return f"По характеристике «{attribute}» у товара «{product}» упоминаний нет."
        months = "; ".join(f"{m}: {percent(neg / total)} негатива из {total} упоминаний"
                           for m, (neg, total) in item.by_month.items())
        return (f"«{attribute}», товар «{product}»: упоминаний {item.mentions}, охват {percent(item.coverage)} отзывов, "
                f"негатив {item.negative}, нейтрально {item.neutral}, позитив {item.positive}, "
                f"доля негатива {percent(item.neg_share)}." + (f" По месяцам: {months}." if months else ""))

    def compare(self, attribute: str) -> str:
        row = next((c for c in self.result.comparisons if c.attribute == attribute), None)
        if row is None or not row.competitors:
            return f"Данных о конкурентах по характеристике «{attribute}» нет."
        parts = [f"{name}: доля негатива {percent(neg)}" for name, (neg, _) in row.competitors.items()]
        verdict = "данных для вывода мало" if row.verdict == NO_DATA else f"наш товар {row.verdict}"
        return f"«{attribute}»: наш товар, доля негатива {percent(row.own_neg_share)}; {'; '.join(parts)}. Вывод: {verdict}."

    def list_issues(self) -> str:
        if not self.result.issues:
            return "Повторяющихся проблем не найдено."
        return "\n".join(f"{i.id}. {i.title} ({i.attribute}): {percent(i.coverage)} отзывов, упоминаний {len(i.sentence_ids)}"
                         for i in self.result.issues)


def answer_question(gateway: LLMGateway, result: PipelineResult, question: str, category: str = "одежда",
                    max_steps: int = MAX_STEPS) -> Answer:
    """Отвечает на вопрос продавца по результатам анализа."""
    client = gateway.client
    tools = Toolbox(result, client.embed if getattr(client, "embed_model", "") else None)
    prompt = load_prompt(PROMPT_VERSION).format(
        product=result.own_product, category=category, question=question.strip(), max_steps=max_steps,
        attributes=", ".join(result.taxonomy), products=", ".join(sorted({r.product for r in result.reviews})))
    dialog = [{"role": "user", "content": prompt}]
    calls: list[tuple[str, dict]] = []

    def check(step: Step) -> list[str]:
        if step.answer is not None and step.answer.strip():
            unknown = sorted(set(step.sentence_ids) - tools.seen)
            if unknown:
                return [f"предложений {unknown} инструменты не возвращали: ссылайся только на полученные"]
            if tools.seen and not step.sentence_ids:
                return ["укажи в sentence_ids номера предложений, на которые опирается ответ"]
            return []
        if step.tool not in TOOLS:
            return [f"укажи инструмент из списка {list(TOOLS)} или окончательный ответ в поле answer"]
        if len(calls) >= max_steps:
            return [f"лимит {max_steps} вызовов исчерпан: дай окончательный ответ в поле answer"]
        return []

    try:
        while True:
            step = gateway.ask_json("qa", dialog, Step, check=check, max_tokens=800)
            if step.answer is not None and step.answer.strip():
                return Answer(step.answer.strip(), list(dict.fromkeys(step.sentence_ids)), calls)
            calls.append((step.tool, step.args))
            observation = tools.call(step.tool, step.args)
            dialog = dialog + [
                {"role": "assistant", "content": step.model_dump_json(exclude_none=True)},
                {"role": "user", "content": f"Результат {step.tool}:\n{observation}"},
            ]
    except (FormatError, LLMError) as error:
        return Answer(f"Не удалось получить ответ: {error}", [], calls, ok=False)
