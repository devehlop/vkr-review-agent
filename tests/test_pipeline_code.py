"""Тесты шагов конвейера без LLM: агрегация, сравнение, приоритизация, проверка опоры, хранилище."""
from datetime import date

import pytest

from review_agent.models import Issue, Recommendation, Review, Sentence
from review_agent.pipeline.aggregate import aggregate
from review_agent.pipeline.compare import BETTER, NO_DATA, WORSE, compare
from review_agent.pipeline.extract import Mention
from review_agent.pipeline.grounding import check_grounding
from review_agent.pipeline.prioritize import prioritize, priority
from review_agent.storage import Storage


def m(attribute, sentiment, feature="", quote="x"):
    return Mention(attribute=attribute, feature=feature, sentiment=sentiment, quote=quote)


@pytest.fixture
def data():
    reviews = [
        Review(1, "свой", rating=2, created=date(2026, 9, 5)),
        Review(2, "свой", rating=5, created=date(2026, 9, 20)),
        Review(3, "свой", rating=1, created=date(2026, 10, 1)),
        Review(4, "свой", rating=4),
        Review(5, "конкурент", rating=5, is_own=False),
    ]
    sentences = [
        Sentence(1, 1, "Швы расходятся."), Sentence(2, 1, "И швы кривые, ткань хорошая."),
        Sentence(3, 2, "Швы ровные."), Sentence(4, 3, "Шов разошёлся сразу."),
        Sentence(5, 4, "Пришло быстро."), Sentence(6, 5, "Швы аккуратные."),
    ]
    mentions = {
        1: [m("Качество пошива", -1, "Швы")],
        2: [m("Качество пошива", -1, "швы"), m("Ткань и материал", 1)],
        3: [m("Качество пошива", 1, "швы")],
        4: [m("Качество пошива", -1, "швы")],
        5: [],
        6: [m("Качество пошива", 1, "швы")],
    }
    return reviews, sentences, mentions


def test_aggregate_counts_and_shares(data):
    stats = {(s.product, s.attribute, s.feature): s for s in aggregate(*data)}
    sewing = stats[("свой", "Качество пошива", "")]
    assert (sewing.mentions, sewing.reviews, sewing.negative, sewing.positive) == (4, 3, 3, 1)
    assert sewing.coverage == 0.75  # 3 отзыва из 4: два упоминания в одном отзыве считаются один раз
    assert sewing.neg_share == 0.75
    assert sewing.avg_rating == pytest.approx(2.67, abs=0.01)
    assert sewing.by_month == {"2026-09": (2, 3), "2026-10": (1, 1)}
    # признак считается без учёта регистра
    assert stats[("свой", "Качество пошива", "швы")].mentions == 4
    assert stats[("конкурент", "Качество пошива", "")].neg_share == 0.0
    assert stats[("свой", "Ткань и материал", "")].coverage == 0.25


def test_aggregate_order_attributes_first(data):
    own = [s for s in aggregate(*data) if s.product == "свой"]
    assert [(s.attribute, s.feature) for s in own][:2] == [("Качество пошива", ""), ("Ткань и материал", "")]


def test_compare_worse_and_no_data(data):
    stats = aggregate(*data)
    rows = {r.attribute: r for r in compare(stats, "свой", min_mentions=1)}
    assert rows["Качество пошива"].verdict == WORSE and rows["Качество пошива"].rarer_at_competitors
    assert rows["Качество пошива"].competitors == {"конкурент": (0.0, 1.0)}
    assert rows["Ткань и материал"].verdict == NO_DATA  # у конкурента нет упоминаний
    # при пороге по числу упоминаний малые выборки не сравниваются
    assert all(r.verdict == NO_DATA for r in compare(stats, "свой", min_mentions=5))


def test_compare_better():
    reviews = [Review(1, "свой"), Review(2, "чужой", is_own=False)]
    sentences = [Sentence(1, 1, "a"), Sentence(2, 2, "b")]
    mentions = {1: [m("Цена", 1)], 2: [m("Цена", -1)]}
    assert compare(aggregate(reviews, sentences, mentions), "свой", min_mentions=1)[0].verdict == BETTER


def test_priority_formula():
    assert priority(0.2, 0.5, 1) == 0.1
    assert priority(0.2, 0.5, 2) == 0.05
    assert priority(0.2, 0.5, 1, rarer_at_competitors=True) == 0.15
    with pytest.raises(ValueError):
        priority(0.2, 0.5, 0)


def test_prioritize_sorts():
    issues = [Issue(1, "свой", "Качество пошива", "Расходятся швы", [1, 2, 4], 0.5, 0.75, True),
              Issue(2, "свой", "Упаковка", "Мнётся", [5, 6, 7], 0.1, 0.5)]
    recs = [Recommendation(1, 2, "Коробка", "упаковка", "", 1), Recommendation(2, 1, "Сменить нитки", "товар", "", 2),
            Recommendation(3, 99, "Без проблемы", "товар", "", 1)]
    ordered = prioritize(recs, issues)
    assert [r.id for r in ordered] == [2, 1, 3]
    assert [r.priority for r in ordered] == [0.2812, 0.05, 0.0]


def test_grounding(data):
    _, sentences, _ = data
    issues = [Issue(1, "свой", "Качество пошива", "Расходятся швы", [1, 2, 4]),
              Issue(2, "свой", "Доставка", "Долго", [5, 77])]
    recs = [
        Recommendation(1, 1, "Усилить швы", "товар", "", 2, quotes=[(1, "швы расходятся"), (4, "Шов разошелся")]),
        Recommendation(2, 1, "Выдумка", "товар", "", 1, quotes=[(1, "нитки гнилые")]),
        Recommendation(3, 2, "Мало опоры", "логистика", "", 1, quotes=[(5, "Пришло быстро")]),
        Recommendation(4, 5, "Нет проблемы", "товар", "", 1, quotes=[(1, "Швы")]),
        Recommendation(5, 1, "Без цитат", "товар", "", 1),
        Recommendation(6, 1, "Чужая цитата", "товар", "", 1, quotes=[(3, "Швы ровные")]),
    ]
    accepted, rejected = check_grounding(recs, issues, sentences)
    assert [r.id for r in accepted] == [1] and accepted[0].grounded
    reasons = {r.id: reason for r, reason in rejected}
    assert "нет в предложении 1" in reasons[2]
    assert "опирается на 1 предложений" in reasons[3]
    assert "несуществующую проблему" in reasons[4]
    assert "нет цитат" in reasons[5]
    assert "не относится к проблеме" in reasons[6]
    assert not any(r.grounded for r, _ in rejected)


def test_storage_roundtrip(data, tmp_path):
    reviews, sentences, mentions = data
    store = Storage(tmp_path / "папка" / "база.db")
    run_id = store.start_run("fake-model", "extract_v1")
    store.save_reviews(run_id, reviews)
    store.save_sentences(run_id, sentences)
    store.save_mentions(run_id, mentions)
    store.save_mentions(run_id, mentions)  # повторное сохранение шага не создаёт дублей
    issues = [Issue(1, "свой", "Качество пошива", "Расходятся швы", [1, 2, 4], 0.5, 0.75, True)]
    recs = [Recommendation(1, 1, "Усилить швы", "товар", "часто жалуются", 2, [(1, "швы расходятся")], 0.28, True)]
    store.save_issues(run_id, issues)
    store.save_recommendations(run_id, recs)
    store.finish_run(run_id)

    assert store.load_reviews(run_id) == reviews
    assert store.load_sentences(run_id) == sentences
    loaded = store.load_mentions(run_id)
    assert {k: v for k, v in mentions.items() if v} == loaded
    assert store.load_issues(run_id) == issues
    assert store.load_recommendations(run_id) == recs
    assert store.run_status(run_id) == "готово"
    # результат, загруженный из базы, даёт ту же статистику
    assert aggregate(store.load_reviews(run_id), store.load_sentences(run_id), loaded) == aggregate(*data)
    # другой запуск не видит чужих данных
    assert store.load_reviews(store.start_run("m", "v")) == []
    store.close()
