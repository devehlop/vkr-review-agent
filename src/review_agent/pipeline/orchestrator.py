"""Оркестратор конвейера: запускает шаги по порядку и собирает результат.

Шаги 1–11 из docs/architecture.md, раздел 3. Шаг 12 (отчёт) — в report/.
Промежуточные результаты сохраняются в SQLite, если передано хранилище.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from review_agent.config import Settings
from review_agent.ingest.loader import ColumnMapping, load_reviews
from review_agent.llm.gateway import LLMGateway
from review_agent.models import AspectStats, Issue, Mention, Product, Recommendation, Review, Sentence
from review_agent.pipeline import compare as compare_step
from review_agent.pipeline.embeddings import Embedder, TfidfEmbedder
from review_agent.pipeline.extract import PROMPT as EXTRACT_PROMPT
from review_agent.pipeline.extract import extract_mentions
from review_agent.pipeline.aggregate import compute_stats
from review_agent.pipeline.grounding import apply_grounding
from review_agent.pipeline.issues import extract_issues
from review_agent.pipeline.prioritize import rank
from review_agent.pipeline.recommend import CritiqueRecord, issue_target, recommend, select_strengths
from review_agent.pipeline.taxonomy import build_taxonomy
from review_agent.preprocess.text import deduplicate, prepare_review, review_sentences
from review_agent.storage.db import Storage
from review_agent.taxonomy import Taxonomy

MIN_REVIEWS_WARNING = 30


@dataclass
class ProductInput:
    product: Product
    path: Path | None = None
    reviews: list[Review] | None = None
    mapping: ColumnMapping | None = None


@dataclass
class AnalysisResult:
    products: list[Product]
    reviews: list[Review]
    sentences: list[Sentence]
    taxonomy: Taxonomy
    mentions: list[Mention] = field(default_factory=list)
    stats: list[AspectStats] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    comparison: list[compare_step.ComparisonRow] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)
    rejected_by_critic: list[Recommendation] = field(default_factory=list)
    ungrounded: list[tuple[Recommendation, list[str]]] = field(default_factory=list)
    critiques: list[CritiqueRecord] = field(default_factory=list)
    skipped_rows: dict[str, list[tuple[int, str]]] = field(default_factory=dict)
    removed_duplicates: int = 0
    failed_sentence_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    llm_summary: dict = field(default_factory=dict)
    run_id: int | None = None

    @property
    def own_ids(self) -> set[str]:
        return {p.id for p in self.products if p.is_own}


def run_analysis(
    inputs: list[ProductInput],
    gateway: LLMGateway,
    settings: Settings | None = None,
    taxonomy: Taxonomy | None = None,
    category: str = "одежда",
    embedder: Embedder | None = None,
    storage: Storage | None = None,
    use_critic: bool = True,
    progress: Callable[[str], None] = lambda _: None,
) -> AnalysisResult:
    settings = settings or Settings()
    embedder = embedder or TfidfEmbedder()
    th = settings.thresholds
    products = [i.product for i in inputs]
    if not any(p.is_own for p in products):
        raise ValueError("Нужен хотя бы один свой товар")

    # Шаг 1. Загрузка
    progress("Загрузка отзывов")
    raw: list[Review] = []
    skipped: dict[str, list[tuple[int, str]]] = {}
    for inp in inputs:
        if inp.reviews is not None:
            raw.extend(inp.reviews)
        elif inp.path is not None:
            loaded = load_reviews(inp.path, inp.product.id, inp.mapping)
            raw.extend(loaded.reviews)
            skipped[inp.product.id] = loaded.skipped
        else:
            raise ValueError(f"Для товара {inp.product.name} нет ни файла, ни отзывов")

    # Шаг 2. Предобработка
    progress("Предобработка")
    reviews, removed = deduplicate([prepare_review(r) for r in raw])
    sentences = [s for r in reviews for s in review_sentences(r)]

    run_id = None
    if storage:
        run_id = storage.start_run(gateway.client.provider, gateway.client.model, ".".join(EXTRACT_PROMPT),
                                   params={"use_critic": use_critic, "batch_size": settings.extract_batch_size})
        gateway.on_record = storage.llm_logger(run_id)
        storage.save_products(run_id, products)
        storage.save_reviews(run_id, reviews)
        storage.save_sentences(run_id, sentences)

    # Шаг 3. Справочник
    if taxonomy is None:
        progress("Построение справочника аспектов")
        taxonomy = build_taxonomy(gateway, reviews, category)

    result = AnalysisResult(products=products, reviews=reviews, sentences=sentences, taxonomy=taxonomy,
                            skipped_rows=skipped, removed_duplicates=removed, run_id=run_id)
    for p in products:
        n = sum(r.product_id == p.id for r in reviews)
        if p.is_own and n < MIN_REVIEWS_WARNING:
            result.warnings.append(f"У товара «{p.name}» всего {n} отзывов: выводы ненадёжны")

    # Шаг 4. Извлечение аспектов
    progress(f"Извлечение аспектов: {len(sentences)} предложений")
    extracted = extract_mentions(gateway, sentences, taxonomy, batch_size=settings.extract_batch_size)
    result.mentions = extracted.mentions
    result.failed_sentence_ids = extracted.failed_sentence_ids
    result.errors.extend(extracted.errors)
    if storage:
        storage.save_mentions(run_id, result.mentions)

    # Шаг 5. Агрегация
    progress("Агрегация")
    result.stats = compute_stats(reviews, sentences, result.mentions)

    # Шаг 6. Проблемы
    progress("Выделение проблем")
    result.issues = extract_issues(gateway, reviews, sentences, result.mentions, embedder, result.own_ids,
                                   min_size=th.min_issue_size)
    if storage:
        storage.save_issues(run_id, result.issues)

    # Шаг 7. Сравнение с конкурентами
    result.comparison = compare_step.compare_attributes(result.stats, result.own_ids, th.competitor_ratio)
    flags = compare_step.competitor_flags(result.issues, result.stats, result.own_ids)

    # Шаги 8–9. Рекомендации и критик
    progress("Рекомендации")
    mention_by_id = {m.id: m for m in result.mentions}
    targets = [issue_target(i, mention_by_id, flags.get(i.id, False)) for i in result.issues]
    product_of_review = {r.id: r.product_id for r in reviews}
    review_of_sentence = {s.id: s.review_id for s in sentences}
    product_of_mention = {m.id: product_of_review[review_of_sentence[m.sentence_id]] for m in result.mentions}
    targets += select_strengths(result.stats, result.mentions, product_of_mention, result.own_ids)
    rec = recommend(gateway, targets, use_critic=use_critic, max_rounds=th.max_critic_rounds)
    result.rejected_by_critic = rec.rejected
    result.critiques = rec.critiques
    result.errors.extend(rec.errors)

    # Шаг 10. Приоритизация
    ranked = rank(rec.accepted, result.issues, flags, th.competitor_k)

    # Шаг 11. Проверка опоры на отзывы
    grounding = apply_grounding(ranked, result.issues, result.mentions, sentences, th.min_support)
    result.recommendations = grounding.accepted
    result.ungrounded = grounding.rejected

    if storage:
        storage.save_recommendations(run_id, result.recommendations + [r for r, _ in result.ungrounded] + result.rejected_by_critic)
        for c in result.critiques:
            storage.save_critique(run_id, c.recommendation_id, c.iteration, c.scores, c.comment, c.passed)
        storage.finish_run(run_id, "done" if not result.errors else "done_with_errors")
    result.llm_summary = gateway.summary()
    progress("Готово")
    return result
