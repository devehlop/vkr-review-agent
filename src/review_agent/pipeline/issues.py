"""Шаг 6 конвейера: выделение проблем (FR-10, FR-11).

Негативные упоминания своего товара группируются внутри каждого атрибута по близости
формулировок, затем модель даёт каждой группе название.
"""
from collections.abc import Callable

import numpy as np
from pydantic import BaseModel
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

from review_agent.config import load_prompt
from review_agent.llm.base import LLMError
from review_agent.llm.gateway import FormatError, LLMGateway
from review_agent.models import Issue, Review, Sentence
from review_agent.pipeline.aggregate import AspectStats
from review_agent.pipeline.compare import Comparison
from review_agent.pipeline.extract import Mention

PROMPT_VERSION = "issue_title_v1"
Embedder = Callable[[list[str]], np.ndarray]


class IssueTitle(BaseModel):
    title: str


def tfidf_embed(texts: list[str]) -> np.ndarray:
    """Векторы по буквенным сочетаниям (TF-IDF). Не требуют модели и интернета.

    Улавливают однокоренные слова («шов», «швы», «швах»), но не синонимы.
    Нейросетевые эмбеддинги подключаются через параметр embed.
    """
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=True)
    return vectorizer.fit_transform([t.replace("ё", "е") for t in texts]).toarray()


def cluster_texts(texts: list[str], embed: Embedder = tfidf_embed, distance_threshold: float = 0.75) -> list[list[int]]:
    """Агломеративная кластеризация по косинусному расстоянию. Число кластеров заранее не задаётся.

    Возвращает группы индексов текстов, сначала самые большие.
    """
    if len(texts) < 2:
        return [list(range(len(texts)))] if texts else []
    vectors = np.asarray(embed(texts), dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.where(norms == 0, 1, norms)
    labels = AgglomerativeClustering(
        n_clusters=None, distance_threshold=distance_threshold, metric="cosine", linkage="average"
    ).fit_predict(vectors)
    groups: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(int(label), []).append(index)
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


def _central(indices: list[int], texts: list[str], embed: Embedder, limit: int) -> list[int]:
    """Индексы текстов, ближайших к центру группы: по ним модель формулирует проблему."""
    if len(indices) <= limit:
        return indices
    vectors = np.asarray(embed([texts[i] for i in indices]), dtype=float)
    distance = np.linalg.norm(vectors - vectors.mean(axis=0), axis=1)
    return [indices[i] for i in np.argsort(distance)[:limit]]


def find_issues(
    gateway: LLMGateway,
    reviews: list[Review],
    sentences: list[Sentence],
    mentions: dict[int, list[Mention]],
    stats: list[AspectStats],
    own_product: str,
    comparisons: list[Comparison] | None = None,
    category: str = "одежда",
    min_size: int = 3,
    distance_threshold: float = 0.75,
    embed: Embedder = tfidf_embed,
    quotes_for_title: int = 10,
) -> list[Issue]:
    """Возвращает проблемы своего товара. Группы меньше min_size упоминаний проблемами не считаются."""
    review_by_id = {r.id: r for r in reviews}
    own_total = sum(r.product == own_product for r in reviews)
    neg_share = {s.attribute: s.neg_share for s in stats if s.product == own_product and not s.feature}
    rarer = {c.attribute: c.rarer_at_competitors for c in comparisons or []}

    # атрибут -> список (предложение, цитата) с негативной тональностью
    negative: dict[str, list[tuple[Sentence, str]]] = {}
    for sentence in sentences:
        review = review_by_id.get(sentence.review_id)
        if review is None or review.product != own_product:
            continue
        for mention in mentions.get(sentence.id, []):
            if mention.sentiment == -1:
                negative.setdefault(mention.attribute, []).append((sentence, mention.quote))

    issues: list[Issue] = []
    template = load_prompt(PROMPT_VERSION)
    for attribute, items in negative.items():
        if len(items) < min_size:
            continue
        quotes = [quote for _, quote in items]
        for group in cluster_texts(quotes, embed, distance_threshold):
            if len(group) < min_size:
                continue
            sentence_ids = sorted({items[i][0].id for i in group})
            review_ids = {items[i][0].review_id for i in group}
            chosen = _central(group, quotes, embed, quotes_for_title)
            prompt = template.format(
                category=category, attribute=attribute, quotes="\n".join(f"- {quotes[i]}" for i in chosen))
            try:
                title = gateway.ask_json(
                    "issue_title", [{"role": "user", "content": prompt}], IssueTitle,
                    check=lambda r: [] if r.title.strip() else ["пустое название"], max_tokens=200,
                ).title.strip()
            except (FormatError, LLMError):
                title = f"{attribute}: {quotes[chosen[0]]}"  # запасной вариант: название по самой типичной цитате
            issues.append(Issue(
                id=len(issues) + 1, product=own_product, attribute=attribute, title=title,
                sentence_ids=sentence_ids,
                coverage=round(len(review_ids) / own_total, 4) if own_total else 0.0,
                neg_share=neg_share.get(attribute, 0.0),
                rarer_at_competitors=rarer.get(attribute, False),
            ))
    issues.sort(key=lambda i: (-i.coverage, i.id))
    for number, issue in enumerate(issues, start=1):
        issue.id = number
    return issues
