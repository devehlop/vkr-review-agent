import json

import pytest

from review_agent.llm.fake import FakeLLMClient
from review_agent.llm.gateway import LLMGateway
from review_agent.models import Mention, Review, Sentence
from review_agent.pipeline.embeddings import TfidfEmbedder
from review_agent.pipeline.issues import cluster_texts, extract_issues, find_clusters

SIZE = "Размер и посадка"
QUOTES = {
    "own": [
        (SIZE, -1, "маломерит"), (SIZE, -1, "сильно маломерит"), (SIZE, -1, "маломерит на размер"),
        (SIZE, -1, "рукава короткие"), (SIZE, -1, "короткие рукава"), (SIZE, -1, "рукава коротковаты"),
        (SIZE, -1, "длинное, до пола"),
        (SIZE, 1, "размер подошёл"),
        ("Цена", -1, "дорого"),
    ],
    "comp": [(SIZE, -1, "маломерит"), (SIZE, -1, "маломерит"), (SIZE, -1, "маломерит")],
}


def make_data():
    reviews, sentences, mentions = [], [], []
    for pid, items in QUOTES.items():
        for k, (attr, sent, quote) in enumerate(items):
            rid = f"{pid}-r{k}"
            reviews.append(Review(id=rid, product_id=pid, text=quote))
            sentences.append(Sentence(id=f"{rid}:0", review_id=rid, text=quote))
            mentions.append(Mention(id=f"{rid}#0", sentence_id=f"{rid}:0", attribute=attr, feature="другое", sentiment=sent, quote=quote))
    return reviews, sentences, mentions


def test_cluster_edge_cases():
    import numpy as np

    assert cluster_texts(np.zeros((0, 3)), 0.5) == []
    assert cluster_texts(np.ones((1, 3)), 0.5) == [0]


def test_find_clusters_separates_topics_and_drops_small():
    reviews, sentences, mentions = make_data()
    clusters = find_clusters(reviews, sentences, mentions, TfidfEmbedder(), product_ids={"own"})
    groups = sorted(sorted(c.representative_quotes) for c in clusters)
    assert groups == [
        ["короткие рукава", "рукава короткие", "рукава коротковаты"],
        ["маломерит", "маломерит на размер", "сильно маломерит"],
    ]
    # «длинное, до пола» и «дорого» — меньше трёх упоминаний, проблемами не становятся
    assert all(c.product_id == "own" for c in clusters)


def test_extract_issues_with_fake_llm():
    reviews, sentences, mentions = make_data()

    def responder(messages):
        text = messages[-1].content
        return json.dumps({"title": "Маломерит на размер" if "маломерит" in text else "Короткие рукава"}, ensure_ascii=False)

    gw = LLMGateway(FakeLLMClient(responder))
    issues = extract_issues(gw, reviews, sentences, mentions, TfidfEmbedder(), own_product_ids={"own"})
    assert sorted(i.title for i in issues) == ["Короткие рукава", "Маломерит на размер"]
    for i in issues:
        assert i.coverage == pytest.approx(3 / 9)
        # По атрибуту «Размер» у своего товара 7 негативных из 8 упоминаний
        assert i.neg_share == pytest.approx(7 / 8)
        assert len(i.mention_ids) == 3
