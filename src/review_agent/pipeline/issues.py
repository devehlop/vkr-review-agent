"""Шаг 6. Выделение проблем (FR-10, FR-11).

1. Берутся негативные упоминания своего товара.
2. Внутри каждого атрибута цитаты кластеризуются по эмбеддингам
   (агломеративная кластеризация, порог по косинусному расстоянию; число
   кластеров заранее не задаётся).
3. Кластеры меньше min_size в проблемы не выделяются.
4. Для каждого кластера модель формулирует проблему по цитатам, ближайшим к центру.

Охват проблемы — доля отзывов товара, где она встречается. Доля негатива
берётся по атрибуту в целом (шаг 5): она показывает, насколько атрибут
в целом воспринимается плохо, и вместе с охватом входит в формулу приоритета.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field

from review_agent.config import Thresholds
from review_agent.llm.gateway import LLMGateway
from review_agent.models import Issue, Mention, Review, Sentence
from review_agent.pipeline.embeddings import Embedder
from review_agent.prompts import render_prompt

PROMPT = ("name_issue", "v1")


@dataclass
class Cluster:
    product_id: str
    attribute: str
    mention_ids: list[str]
    # Цитаты по близости к центру кластера, первая — самая типичная
    representative_quotes: list[str]


class IssueTitle(BaseModel):
    title: str = Field(min_length=3, max_length=120)


def cluster_texts(vectors: np.ndarray, distance_threshold: float) -> list[int]:
    """Номер кластера для каждого вектора."""
    n = len(vectors)
    if n == 0:
        return []
    if n == 1:
        return [0]
    from sklearn.cluster import AgglomerativeClustering

    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    return model.fit_predict(vectors).tolist()


def find_clusters(
    reviews: list[Review],
    sentences: list[Sentence],
    mentions: list[Mention],
    embedder: Embedder,
    product_ids: set[str] | None = None,
    distance_threshold: float | None = None,
    min_size: int = Thresholds.min_issue_size,
    top_quotes: int = 7,
) -> list[Cluster]:
    """Кластеры негативных упоминаний по каждому товару и атрибуту, крупные — первыми."""
    product_of_review = {r.id: r.product_id for r in reviews}
    product_of_sentence = {s.id: product_of_review[s.review_id] for s in sentences}

    groups: dict[tuple[str, str], list[Mention]] = defaultdict(list)
    for m in mentions:
        pid = product_of_sentence[m.sentence_id]
        if m.sentiment == -1 and (product_ids is None or pid in product_ids):
            groups[(pid, m.attribute)].append(m)

    if distance_threshold is None:
        distance_threshold = embedder.default_threshold
    clusters: list[Cluster] = []
    for (pid, attribute), items in sorted(groups.items()):
        if len(items) < min_size:
            continue
        vectors = embedder.embed([m.quote for m in items])
        labels = cluster_texts(vectors, distance_threshold)
        for label in sorted(set(labels)):
            idx = [i for i, lab in enumerate(labels) if lab == label]
            if len(idx) < min_size:
                continue
            centroid = vectors[idx].mean(axis=0)
            order = sorted(idx, key=lambda i: -float(vectors[i] @ centroid))
            quotes: list[str] = []
            for i in order:
                if items[i].quote not in quotes:
                    quotes.append(items[i].quote)
            clusters.append(
                Cluster(
                    product_id=pid,
                    attribute=attribute,
                    mention_ids=[items[i].id for i in order],
                    representative_quotes=quotes[:top_quotes],
                )
            )
    return sorted(clusters, key=lambda c: (-len(c.mention_ids), c.product_id, c.attribute))


def name_cluster(gateway: LLMGateway, cluster: Cluster) -> str:
    quotes = "\n".join(f"- {q}" for q in cluster.representative_quotes)
    messages = render_prompt(*PROMPT, attribute=cluster.attribute, quotes=quotes)

    def validate(t: IssueTitle) -> list[str]:
        words = len(t.title.split())
        return [] if 2 <= words <= 12 else [f"формулировка должна быть 3–10 слов, сейчас {words}"]

    return gateway.structured("issues", messages, IssueTitle, validate).title.strip()


def build_issues(
    clusters: list[Cluster],
    titles: list[str],
    reviews: list[Review],
    sentences: list[Sentence],
    mentions: list[Mention],
) -> list[Issue]:
    """Собирает проблемы с охватом и долей негатива по атрибуту."""
    review_of_sentence = {s.id: s.review_id for s in sentences}
    product_of_review = {r.id: r.product_id for r in reviews}
    totals: dict[str, int] = defaultdict(int)
    for r in reviews:
        totals[r.product_id] += 1
    mention_by_id = {m.id: m for m in mentions}

    attr_counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for m in mentions:
        pid = product_of_review[review_of_sentence[m.sentence_id]]
        attr_counts[(pid, m.attribute)][0] += m.sentiment == -1
        attr_counts[(pid, m.attribute)][1] += 1

    issues = []
    for k, (cluster, title) in enumerate(zip(clusters, titles, strict=True), start=1):
        review_ids = {review_of_sentence[mention_by_id[mid].sentence_id] for mid in cluster.mention_ids}
        neg, total = attr_counts[(cluster.product_id, cluster.attribute)]
        issues.append(
            Issue(
                id=f"I{k}",
                product_id=cluster.product_id,
                attribute=cluster.attribute,
                title=title,
                mention_ids=cluster.mention_ids,
                coverage=len(review_ids) / totals[cluster.product_id],
                neg_share=neg / total if total else 0.0,
            )
        )
    return issues


def extract_issues(
    gateway: LLMGateway,
    reviews: list[Review],
    sentences: list[Sentence],
    mentions: list[Mention],
    embedder: Embedder,
    own_product_ids: set[str],
    distance_threshold: float | None = None,
    min_size: int = Thresholds.min_issue_size,
) -> list[Issue]:
    """Весь шаг 6: кластеры своих товаров → формулировки → проблемы."""
    clusters = find_clusters(reviews, sentences, mentions, embedder, own_product_ids, distance_threshold, min_size)
    titles = [name_cluster(gateway, c) for c in clusters]
    return build_issues(clusters, titles, reviews, sentences, mentions)
