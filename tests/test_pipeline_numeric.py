from datetime import date

import pytest

from review_agent.models import AspectStats, Issue, Mention, Recommendation, RecommendationType, Review, Sentence
from review_agent.pipeline.aggregate import compute_stats, monthly_negative_share
from review_agent.pipeline.grounding import apply_grounding
from review_agent.pipeline.prioritize import competitor_is_better, priority, rank

FAB = "Ткань и материал"

REVIEWS = [
    Review(id="r1", product_id="own", text="x", rating=2, created=date(2026, 9, 3)),
    Review(id="r2", product_id="own", text="x", rating=4, created=date(2026, 10, 1)),
    Review(id="r3", product_id="own", text="x", rating=5, created=date(2026, 10, 9)),
    Review(id="r4", product_id="own", text="x"),
]
SENTS = [
    Sentence(id="s1", review_id="r1", text="Ткань тонкая и колючая"),
    Sentence(id="s2", review_id="r2", text="Ткань тонковата"),
    Sentence(id="s3", review_id="r3", text="Ткань приятная"),
    Sentence(id="s4", review_id="r1", text="Ткань просвечивает"),
]
MENTIONS = [
    Mention(id="m1", sentence_id="s1", attribute=FAB, feature="толщина и плотность", sentiment=-1, quote="Ткань тонкая"),
    Mention(id="m2", sentence_id="s2", attribute=FAB, feature="толщина и плотность", sentiment=-1, quote="Ткань тонковата"),
    Mention(id="m3", sentence_id="s3", attribute=FAB, feature="ощущение на теле", sentiment=1, quote="Ткань приятная"),
    Mention(id="m4", sentence_id="s4", attribute=FAB, feature="прозрачность", sentiment=-1, quote="просвечивает"),
]


def test_compute_stats():
    [st] = compute_stats(REVIEWS, SENTS, MENTIONS)
    assert st.mentions == 4 and st.reviews_with == 3
    assert st.coverage == pytest.approx(3 / 4)
    assert st.neg_share == pytest.approx(3 / 4)
    assert st.avg_rating == pytest.approx((2 + 4 + 5) / 3)
    by_feature = compute_stats(REVIEWS, SENTS, MENTIONS, by_feature=True)
    assert {(s.feature, s.mentions) for s in by_feature} == {
        ("толщина и плотность", 2), ("ощущение на теле", 1), ("прозрачность", 1),
    }


def test_monthly_negative_share():
    assert monthly_negative_share(REVIEWS, SENTS, MENTIONS, "own", FAB) == {"2026-09": 1.0, "2026-10": 0.5}


def _stats(pid, coverage, neg):
    return AspectStats(product_id=pid, attribute=FAB, mentions=10, reviews_with=5, coverage=coverage, neg_share=neg, pos_share=0)


def test_competitor_is_better():
    own = _stats("own", 0.4, 0.5)  # негатив в 20 % отзывов
    assert competitor_is_better(own, [_stats("c1", 0.2, 0.25), _stats("c2", 0.1, 0.5)])  # в среднем 5 %
    assert not competitor_is_better(own, [_stats("c1", 0.4, 0.5)])
    assert not competitor_is_better(own, [])


def test_priority_formula_and_rank():
    assert priority(0.4, 0.5, 2, competitor_better=True, k=1.5) == pytest.approx(0.15)
    issues = [
        Issue(id="i1", product_id="own", attribute=FAB, title="Тонкая ткань", coverage=0.4, neg_share=0.5),
        Issue(id="i2", product_id="own", attribute="Размер и посадка", title="Маломерит", coverage=0.3, neg_share=0.9),
    ]
    recs = [
        Recommendation(id="a", issue_id="i1", action="Сменить ткань", type=RecommendationType.PRODUCT, rationale="", effort=3),
        Recommendation(id="b", issue_id="i2", action="Поправить сетку", type=RecommendationType.SIZE_CHART, rationale="", effort=1),
        Recommendation(id="c", issue_id=None, action="Подчеркнуть цвет", type=RecommendationType.CARD, rationale="", effort=1),
    ]
    ranked = rank(recs, issues, {"i1": True})
    assert [r.id for r in ranked] == ["b", "a", "c"]
    assert ranked[0].priority == pytest.approx(0.27)


def test_grounding_rejects_weak_issue_and_invented_quote():
    issues = [
        Issue(id="ok", product_id="own", attribute=FAB, title="Тонкая ткань", mention_ids=["m1", "m2", "m4"]),
        Issue(id="weak", product_id="own", attribute=FAB, title="Колючая", mention_ids=["m1"]),
    ]
    recs = [
        Recommendation(id="good", issue_id="ok", action="Плотнее ткань", type=RecommendationType.PRODUCT,
                       rationale="", effort=2, quotes=["ткань тонковата"]),
        Recommendation(id="fake_quote", issue_id="ok", action="...", type=RecommendationType.PRODUCT,
                       rationale="", effort=2, quotes=["ткань рвётся"]),
        Recommendation(id="weak_issue", issue_id="weak", action="...", type=RecommendationType.PRODUCT,
                       rationale="", effort=1, quotes=["Ткань тонкая"]),
        Recommendation(id="no_issue", issue_id="missing", action="...", type=RecommendationType.PRODUCT,
                       rationale="", effort=1, quotes=["Ткань тонкая"]),
    ]
    rep = apply_grounding(recs, issues, MENTIONS, SENTS, min_support=3)
    assert [r.id for r in rep.accepted] == ["good"] and rep.accepted[0].grounded
    reasons = {r.id: p for r, p in rep.rejected}
    assert "не найдена" in reasons["fake_quote"][0]
    assert "weak" in rep.bad_issues and "не прошла проверку" in reasons["weak_issue"][0]
    assert "не существует" in reasons["no_issue"][0]
