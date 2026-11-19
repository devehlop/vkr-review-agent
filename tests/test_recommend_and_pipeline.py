import json

from review_agent.llm.demo import demo_client
from review_agent.llm.fake import FakeLLMClient
from review_agent.llm.gateway import LLMGateway
from review_agent.models import AspectStats, Issue, Mention, Product, Review
from review_agent.pipeline.compare import compare_attributes
from review_agent.pipeline.orchestrator import ProductInput, run_analysis
from review_agent.pipeline.recommend import issue_target, recommend
from review_agent.report.markdown import render_markdown
from review_agent.storage.db import Storage
from review_agent.taxonomy import load_taxonomy

SIZE = "Размер и посадка"
MENTIONS = {f"m{k}": Mention(id=f"m{k}", sentence_id=f"s{k}", attribute=SIZE, feature="маломерит", sentiment=-1, quote=q)
            for k, q in enumerate(["маломерит", "сильно маломерит", "маломерит на размер"])}
ISSUE = Issue(id="I1", product_id="own", attribute=SIZE, title="Маломерит на размер", mention_ids=list(MENTIONS), coverage=0.3, neg_share=0.8)

REC = {"recommendations": [{"action": "Указать в карточке совет брать на размер больше", "type": "карточка товара",
                            "rationale": "30 % отзывов жалуются на маломерность", "effort": 1, "quotes": ["маломерит"]}]}


def answer(scores):
    return json.dumps({"scores": dict(zip(["specificity", "grounding", "feasibility", "novelty"], scores)), "comment": "уточни замеры"})


def test_critic_returns_for_rework_then_accepts():
    client = FakeLLMClient([json.dumps(REC, ensure_ascii=False), answer([2, 5, 5, 5]),
                            json.dumps(REC, ensure_ascii=False), answer([5, 5, 5, 5])])
    res = recommend(LLMGateway(client, use_cache=False), [issue_target(ISSUE, MENTIONS, competitor_better=False)])
    assert [r.id for r in res.accepted] == ["I1-R2"]
    assert [c.passed for c in res.critiques] == [False, True]
    # Во втором запросе генератор видит замечания критика
    assert "уточни замеры" in client.calls[2][-1].content


def test_critic_rejects_after_max_rounds():
    client = FakeLLMClient(lambda m: answer([1, 1, 1, 1]) if "рецензент" in m[0].content else json.dumps(REC, ensure_ascii=False))
    res = recommend(LLMGateway(client, use_cache=False), [issue_target(ISSUE, MENTIONS, False)], max_rounds=2)
    assert not res.accepted and len(res.rejected) == 1 and len(res.critiques) == 3


def test_without_critic():
    client = FakeLLMClient([json.dumps(REC, ensure_ascii=False)])
    res = recommend(LLMGateway(client), [issue_target(ISSUE, MENTIONS, False)], use_critic=False)
    assert len(res.accepted) == 1 and not res.critiques


def test_generator_quote_must_come_from_reviews():
    bad = json.loads(json.dumps(REC))
    bad["recommendations"][0]["quotes"] = ["платье рвётся"]
    client = FakeLLMClient([json.dumps(bad, ensure_ascii=False), json.dumps(REC, ensure_ascii=False), answer([5, 5, 5, 5])])
    res = recommend(LLMGateway(client), [issue_target(ISSUE, MENTIONS, False)])
    assert len(res.accepted) == 1
    assert "цитаты «платье рвётся» нет в списке" in client.calls[1][-1].content


def _st(pid, attr, cov, neg):
    return AspectStats(product_id=pid, attribute=attr, mentions=5, reviews_with=5, coverage=cov, neg_share=neg, pos_share=1 - neg)


def test_compare_attributes():
    stats = [_st("own", SIZE, 0.5, 0.8), _st("c1", SIZE, 0.2, 0.5), _st("own", "Цена", 0.1, 0.2), _st("c1", "Цена", 0.4, 0.5),
             _st("own", "Доставка", 0.2, 0.5)]
    rows = {r.attribute: r for r in compare_attributes(stats, {"own"})}
    assert rows[SIZE].verdict == "хуже"
    assert rows["Цена"].verdict == "лучше"
    # У конкурента нет жалоб на доставку — значит, свой товар хуже
    assert rows["Доставка"].verdict == "хуже" and rows["Доставка"].competitor_rate == 0


def _reviews(pid, texts):
    return [Review(id=f"{pid}-{k}", product_id=pid, text=t, rating=2 if "маломер" in t else 5) for k, t in enumerate(texts)]


def test_full_pipeline_on_demo_client(tmp_path):
    own = _reviews("own", ["Маломерит сильно, брала M.", "Маломерит на размер.", "Очень маломерит.", "Маломерит, обидно.",
                           "Ткань мягкая и приятная.", "Ткань мягкая.", "Мягкая ткань, спасибо.", "Заказывала маме."])
    comp = _reviews("comp", ["Ткань мягкая.", "Хорошее платье, рекомендую.", "Размер подошёл отлично."])
    db = Storage(tmp_path / "run.db")
    result = run_analysis(
        [ProductInput(Product(id="own", name="Платье"), reviews=own),
         ProductInput(Product(id="comp", name="Платье конкурента", is_own=False), reviews=comp)],
        LLMGateway(demo_client()),
        taxonomy=None,
        storage=db,
    )
    assert result.taxonomy.version.startswith("auto-")
    assert [i.attribute for i in result.issues] == [SIZE]
    assert result.recommendations and all(r.grounded for r in result.recommendations)
    assert any(r.issue_id is None for r in result.recommendations)  # сильная сторона: ткань
    assert result.warnings  # меньше 30 отзывов
    md = render_markdown(result)
    for section in ("## Кратко", "## Рекомендации по приоритету", "## Проблемы", "## Сравнение с конкурентами"):
        assert section in md
    assert db.run_cost(result.run_id)["calls"] > 0
    status = db.conn.execute("SELECT status FROM run").fetchone()[0]
    assert status == "done"


def test_pipeline_with_fixed_taxonomy_needs_no_taxonomy_call():
    client = demo_client()
    run_analysis([ProductInput(Product(id="own", name="Платье"), reviews=_reviews("own", ["Маломерит."] * 1))],
                 LLMGateway(client), taxonomy=load_taxonomy())
    assert not any("Строишь справочник" in c[0].content for c in client.calls)
