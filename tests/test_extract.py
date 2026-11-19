import json

from review_agent.llm.fake import FakeLLMClient
from review_agent.llm.gateway import LLMGateway
from review_agent.models import Mention, Sentence
from review_agent.pipeline.extract import (
    ExtractedMention,
    FewShotExample,
    build_messages,
    extract_mentions,
    mentions_to_labels,
)
from review_agent.taxonomy import load_taxonomy

TAX = load_taxonomy()
SENTS = [
    Sentence(id="r1:0", review_id="r1", text="Ткань тонкая, но цвет красивый"),
    Sentence(id="r2:0", review_id="r2", text="Заказывала маме"),
]
GOOD = {"results": [
    {"id": "1", "mentions": [
        {"attribute": "Ткань и материал", "feature": "толщина и плотность", "sentiment": -1, "quote": "Ткань тонкая"},
        {"attribute": "Внешний вид и цвет", "feature": "цвет", "sentiment": 1, "quote": "цвет красивый"},
    ]},
    {"id": "2", "mentions": []},
]}


def test_prompt_contains_taxonomy_examples_and_sentences():
    ex = [FewShotExample("Маломерит", [ExtractedMention(attribute="Размер и посадка", feature="маломерит", sentiment=-1, quote="Маломерит")])]
    msgs = build_messages(SENTS, TAX, ex)
    assert msgs[0].role == "system"
    user = msgs[-1].content
    assert "- Размер и посадка: маломерит" in user
    assert "Предложение: Маломерит" in user
    assert "1. Ткань тонкая, но цвет красивый" in user and "2. Заказывала маме" in user
    assert "$" not in user


def test_extract_happy_path():
    gw = LLMGateway(FakeLLMClient([json.dumps(GOOD, ensure_ascii=False)]))
    res = extract_mentions(gw, SENTS, TAX)
    assert not res.failed_sentence_ids
    assert [(m.sentence_id, m.attribute, m.sentiment) for m in res.mentions] == [
        ("r1:0", "Ткань и материал", -1),
        ("r1:0", "Внешний вид и цвет", 1),
    ]
    assert mentions_to_labels(res.mentions, TAX) == {"r1:0": {"fabric": -1, "look": 1}}


def test_extract_retries_on_invented_quote_and_attribute():
    bad = {"results": [
        {"id": "1", "mentions": [{"attribute": "Качество", "sentiment": -1, "quote": "ужасная ткань"}]},
    ]}
    client = FakeLLMClient([json.dumps(bad, ensure_ascii=False), json.dumps(GOOD, ensure_ascii=False)])
    gw = LLMGateway(client)
    res = extract_mentions(gw, SENTS, TAX)
    assert len(res.mentions) == 2
    feedback = client.calls[1][-1].content
    assert "нет ответа для предложений: 2" in feedback
    assert "атрибута «Качество» нет в справочнике" in feedback
    assert "цитаты «ужасная ткань» нет" in feedback


def test_failed_batch_is_skipped_not_fatal():
    gw = LLMGateway(FakeLLMClient(lambda _: "не знаю"), max_format_retries=1)
    res = extract_mentions(gw, SENTS, TAX, batch_size=1)
    assert res.failed_sentence_ids == ["r1:0", "r2:0"]
    assert len(res.errors) == 2


def test_mixed_sentiment_becomes_neutral():
    ms = [
        Mention(sentence_id="s", attribute="Ткань и материал", feature="другое", sentiment=1, quote="a"),
        Mention(sentence_id="s", attribute="Ткань и материал", feature="другое", sentiment=-1, quote="b"),
        Mention(sentence_id="s", attribute="Цена", feature="другое", sentiment=0, quote="c"),
        Mention(sentence_id="s", attribute="Цена", feature="другое", sentiment=-1, quote="d"),
    ]
    assert mentions_to_labels(ms, TAX) == {"s": {"fabric": 0, "price": -1}}


def test_fewshot_examples_file_is_valid_and_not_from_eval_splits():
    import csv
    from pathlib import Path

    from review_agent.pipeline.extract import quote_in_sentence

    data = json.loads(Path("eval/fewshot_examples.json").read_text(encoding="utf-8"))
    eval_sentences = set()
    for split in ("mini", "test"):
        with open(f"data/annotation/{split}.csv", encoding="utf-8") as f:
            eval_sentences |= {r["sentence"] for r in csv.DictReader(f)}
    for ex in data["examples"]:
        assert ex["sentence"] not in eval_sentences
        for m in ex["mentions"]:
            m = ExtractedMention(**m)
            assert TAX.has(m.attribute) and (m.feature == "другое" or TAX.has(m.attribute, m.feature))
            assert quote_in_sentence(m.quote, ex["sentence"])
