"""Оркестратор: запускает шаги конвейера по порядку, от отзывов к рекомендациям."""
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from review_agent.llm.base import LLMError
from review_agent.llm.gateway import FormatError, LLMGateway, Usage
from review_agent.models import CleanReport, Critique, Issue, Recommendation, Review, Sentence
from review_agent.pipeline import critic as critic_step
from review_agent.pipeline import extract as extract_step
from review_agent.pipeline.aggregate import AspectStats, aggregate
from review_agent.pipeline.compare import Comparison, compare
from review_agent.pipeline.extract import Mention, extract_mentions
from review_agent.pipeline.grounding import check_grounding
from review_agent.pipeline.issues import Embedder, find_issues, tfidf_embed
from review_agent.pipeline.prioritize import prioritize
from review_agent.pipeline.recommend import generate, issue_context
from review_agent.pipeline.taxonomy import build_taxonomy
from review_agent.preprocess import clean_reviews, split_reviews
from review_agent.storage import Storage

MIN_REVIEWS = 30  # при меньшем числе отзывов отчёт выдаётся с предупреждением (ТЗ, 4.5)


@dataclass
class Settings:
    category: str = "одежда"
    use_critic: bool = True  # False: запуск без критика для оценки его вклада (ТЗ, 7.2)
    batch_size: int = 20
    recommendations_per_issue: int = 2
    max_revisions: int = 2
    critic_threshold: int = 4
    min_issue_size: int = 3
    distance_threshold: float = 0.75
    min_mentions_to_compare: int = 5
    compare_threshold: float = 0.10


@dataclass
class PipelineResult:
    own_product: str
    taxonomy: dict[str, str]
    clean: CleanReport
    reviews: list[Review]
    sentences: list[Sentence]
    mentions: dict[int, list[Mention]]
    stats: list[AspectStats]
    comparisons: list[Comparison]
    issues: list[Issue]
    recommendations: list[Recommendation]  # прошли все проверки, по убыванию приоритета
    rejected: list[tuple[str, str]] = field(default_factory=list)  # (действие, причина): видны только в журнале
    critiques: list[Critique] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    usage: Usage | None = None
    run_id: int | None = None


def run_pipeline(
    gateway: LLMGateway,
    reviews: list[Review],
    own_product: str,
    taxonomy: dict[str, str] | None = None,
    settings: Settings | None = None,
    storage: Storage | None = None,
    embed: Embedder = tfidf_embed,
    on_step: Callable[[str], None] | None = None,
) -> PipelineResult:
    """Полный прогон: reviews содержит отзывы на свой товар и, если есть, на товары конкурентов.

    taxonomy: готовый справочник (название атрибута -> пояснение). Если не задан, его составляет модель.
    """
    cfg = settings or Settings()
    warnings: list[str] = []
    say = on_step or (lambda text: None)  # интерфейс показывает, какой шаг выполняется

    # Шаг 2. Предобработка. Отзывы из разных файлов получают сквозную нумерацию.
    say("Предобработка отзывов")
    reviews = [replace(r, id=number) for number, r in enumerate(reviews, start=1)]
    reviews, clean = clean_reviews(reviews)
    own_count = sum(r.product == own_product for r in reviews)
    if own_count == 0:
        raise ValueError(f"Нет отзывов на товар «{own_product}»")
    if own_count < MIN_REVIEWS:
        warnings.append(f"Отзывов на товар всего {own_count}: выводы ненадёжны, нужно не меньше {MIN_REVIEWS}.")
    sentences = split_reviews(reviews)

    run_id = None
    if storage:
        run_id = storage.start_run(gateway.client.model, extract_step.PROMPT_VERSION)
        storage.save_reviews(run_id, reviews)
        storage.save_sentences(run_id, sentences)

    # Шаг 3. Справочник аспектов
    if taxonomy is None:
        say("Составление справочника аспектов")
        taxonomy = build_taxonomy(gateway, reviews, cfg.category).as_hints()

    # Шаг 4. Извлечение аспектов
    say(f"Извлечение аспектов: {len(sentences)} предложений")
    extraction = extract_mentions(gateway, sentences, taxonomy, cfg.category, cfg.batch_size)
    mentions = extraction.mentions
    if extraction.failed_ids:
        warnings.append(f"Не обработано предложений: {len(extraction.failed_ids)} из {len(sentences)} (ошибка модели).")
    if storage:
        storage.save_mentions(run_id, mentions)

    # Шаги 5 и 7. Агрегация и сравнение с конкурентами (без модели)
    stats = aggregate(reviews, sentences, mentions)
    comparisons = compare(stats, own_product, cfg.min_mentions_to_compare, cfg.compare_threshold)
    comparison_by_attribute = {c.attribute: c for c in comparisons}

    # Шаг 6. Проблемы
    say("Выделение проблем")
    issues = find_issues(gateway, reviews, sentences, mentions, stats, own_product, comparisons,
                         cfg.category, cfg.min_issue_size, cfg.distance_threshold, embed)

    # Шаги 8 и 9. Рекомендации и критик
    recommendations: list[Recommendation] = []
    rejected: list[tuple[str, str]] = []
    critiques: list[Critique] = []
    next_id = 1
    for issue in issues:
        say(f"Рекомендации: проблема {issue.id} из {len(issues)}")
        context = issue_context(issue, sentences, comparison_by_attribute.get(issue.attribute), cfg.category)
        try:
            drafts = generate(gateway, context, cfg.recommendations_per_issue)
        except (FormatError, LLMError) as error:
            warnings.append(f"Проблема «{issue.title}»: рекомендации не получены ({error}).")
            continue
        for draft in drafts:
            rec_id, next_id = next_id, next_id + 1
            if cfg.use_critic:
                reviewed = critic_step.review_draft(
                    gateway, context, draft, [r.action for r in recommendations],
                    cfg.max_revisions, cfg.critic_threshold)
                critiques.extend(Critique(rec_id, v.iteration, v.scores, v.comment, v.passed) for v in reviewed.verdicts)
                draft = reviewed.draft
                if not reviewed.accepted:
                    last = reviewed.verdicts[-1]
                    rejected.append((draft.action, f"не прошла проверку критика: {last.comment or last.scores}"))
                    continue
            recommendations.append(Recommendation(
                id=rec_id, issue_id=issue.id, action=draft.action.strip(), type=draft.type,
                rationale=draft.rationale.strip(), effort=draft.effort,
                quotes=[(q.sentence_id, q.quote) for q in draft.quotes],
                expected_effect=draft.expected_effect.strip()))

    # Шаги 10 и 11. Приоритеты и проверка опоры на отзывы (без модели)
    say("Проверка опоры на отзывы и расстановка приоритетов")
    recommendations, ungrounded = check_grounding(recommendations, issues, sentences, cfg.min_issue_size)
    rejected.extend((item.action, f"нет опоры на отзывы: {reason}") for item, reason in ungrounded)
    recommendations = prioritize(recommendations, issues)

    if storage:
        storage.save_issues(run_id, issues)
        storage.save_recommendations(run_id, recommendations)
        storage.save_critiques(run_id, critiques)
        storage.finish_run(run_id)

    return PipelineResult(own_product, taxonomy, clean, reviews, sentences, mentions, stats, comparisons,
                          issues, recommendations, rejected, critiques, warnings, gateway.usage, run_id)
