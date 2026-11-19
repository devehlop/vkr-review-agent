from datetime import date

from review_agent.llm.fake import FakeLLMClient
from review_agent.llm.gateway import LLMGateway
from review_agent.llm.base import Message
from review_agent.models import Mention, Product, Recommendation, RecommendationType, Review, Sentence
from review_agent.storage.db import Storage


def test_roundtrip_and_llm_journal(tmp_path):
    db = Storage(tmp_path / "t.db")
    run = db.start_run("fake", "fake-1", "extract_aspects.v1", taxonomy="clothing_v1", params={"batch": 25})
    db.save_products(run, [Product(id="own", name="Свитер")])
    db.save_reviews(run, [Review(id="r1", product_id="own", text="Ткань тонкая", rating=3, created=date(2026, 10, 1))])
    db.save_sentences(run, [Sentence(id="r1:0", review_id="r1", text="Ткань тонкая")])
    m = Mention(id="m1", sentence_id="r1:0", attribute="Ткань и материал", feature="другое", sentiment=-1, quote="Ткань тонкая")
    db.save_mentions(run, [m])
    db.save_recommendations(run, [Recommendation(id="a", action="x", type=RecommendationType.CARD, rationale="y", effort=1, quotes=["Ткань тонкая"])])

    client = FakeLLMClient(["не json", '{"a": 1}'])
    client.price_in_per_1k = 1.0
    gw = LLMGateway(client, on_record=db.llm_logger(run))
    from pydantic import BaseModel

    class A(BaseModel):
        a: int

    gw.structured("extract", [Message("user", "привет")], A)
    db.finish_run(run)

    assert db.load_mentions(run) == [m]
    assert db.load_sentences(run)[0].text == "Ткань тонкая"
    cost = db.run_cost(run)
    assert cost["calls"] == 2 and cost["errors"] == 1 and cost["cost"] > 0
    status = db.conn.execute("SELECT status FROM run WHERE id = ?", (run,)).fetchone()[0]
    assert status == "done"
    db.close()
