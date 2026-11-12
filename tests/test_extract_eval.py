"""Тесты шага извлечения аспектов, метрик и прогона эксперимента на фиктивной модели."""
import json

import pytest

from eval.metrics import aspect_metrics
from eval.run_experiment import run_model, summary_table
from eval.sample import build_sample, parse_labels, read_sample, write_sample
from review_agent.llm import FakeClient, LLMGateway
from review_agent.models import Sentence
from review_agent.pipeline.extract import build_messages, extract_mentions

TAXONOMY = {"Размер и посадка": "размер", "Ткань и материал": "ткань", "Качество пошива": "швы"}
SENTENCES = [
    Sentence(1, 1, "Ткань приятная, но швы расходятся."),
    Sentence(2, 1, "Заказывала в подарок."),
    Sentence(3, 2, "Сильно маломерит."),
]
GOOD = {"results": [
    {"id": 1, "mentions": [
        {"attribute": "Ткань и материал", "feature": "ткань", "sentiment": 1, "quote": "Ткань приятная"},
        {"attribute": "Качество пошива", "feature": "швы", "sentiment": -1, "quote": "швы расходятся"}]},
    {"id": 2, "mentions": []},
    {"id": 3, "mentions": [{"attribute": "Размер и посадка", "feature": "", "sentiment": -1, "quote": "маломерит"}]},
]}


def test_prompt_contains_taxonomy_and_sentences():
    content = build_messages(SENTENCES, TAXONOMY, "одежда")[0]["content"]
    assert "- Качество пошива: швы" in content and "Сильно маломерит." in content
    assert "{" + "category}" not in content


def test_extract_mentions():
    gateway = LLMGateway(FakeClient([json.dumps(GOOD, ensure_ascii=False)]))
    result = extract_mentions(gateway, SENTENCES, TAXONOMY)
    assert [m.attribute for m in result.mentions[1]] == ["Ткань и материал", "Качество пошива"]
    assert result.mentions[2] == [] and result.failed_ids == []


def test_invented_quote_and_unknown_attribute_are_rejected():
    bad = json.loads(json.dumps(GOOD))
    bad["results"][2]["mentions"][0].update(attribute="Скорость", quote="очень маленький размер")
    client = FakeClient([json.dumps(bad, ensure_ascii=False), json.dumps(GOOD, ensure_ascii=False)])
    gateway = LLMGateway(client)
    result = extract_mentions(gateway, SENTENCES, TAXONOMY)
    assert gateway.usage.format_errors == 1 and result.failed_ids == []
    remark = client.calls[1][-1]["content"]
    assert "нет в справочнике" in remark and "нет в тексте предложения" in remark


def test_failed_batch_is_skipped_and_marked():
    gateway = LLMGateway(FakeClient(["это не JSON"]), format_retries=1)
    result = extract_mentions(gateway, SENTENCES, TAXONOMY, batch_size=2)
    assert result.failed_ids == [1, 2, 3] and result.mentions == {}


def test_metrics():
    gold = {1: {"Ткань и материал": 1, "Качество пошива": -1}, 2: {}, 3: {"Размер и посадка": -1}}
    predicted = {1: {"Ткань и материал": 1, "Размер и посадка": 0}, 2: {}, 3: {"Размер и посадка": 1}}
    m = aspect_metrics(gold, predicted)
    assert (m["пар_в_эталоне"], m["пар_у_модели"], m["совпало"]) == (3, 3, 2)
    assert m["precision"] == pytest.approx(0.667, abs=1e-3) and m["recall"] == pytest.approx(0.667, abs=1e-3)
    assert m["тональность_accuracy"] == 0.5
    assert (m["пустых_в_эталоне"], m["пустых_верно"]) == (1, 1)


def test_metrics_perfect():
    gold = {1: {"Ткань и материал": 1}, 2: {}}
    m = aspect_metrics(gold, gold)
    assert m["f1"] == 1.0 and m["тональность_macro_f1"] == 1.0


def test_parse_labels():
    names = list(TAXONOMY)
    assert parse_labels("", names, 1) is None
    assert parse_labels("-", names, 1) == {}
    assert parse_labels("Размер и посадка=-1; Ткань и материал = +1", names, 1) == {"Размер и посадка": -1, "Ткань и материал": 1}
    with pytest.raises(ValueError, match="нет в справочнике"):
        parse_labels("Цена=1", names, 5)
    with pytest.raises(ValueError, match="должна быть"):
        parse_labels("Размер и посадка=2", names, 5)


def test_sample_roundtrip_is_reproducible(tmp_path):
    reviews = [(f"Отзыв номер {i} про хорошую ткань платья. А вот швы номер {i} совсем плохие.", label)
               for i in range(30) for label in ("negative", "positive")]
    rows = build_sample(reviews, size=10, seed=1)
    assert rows == build_sample(reviews, size=10, seed=1)
    assert len(rows) == 10 and len({r["предложение"] for r in rows}) == 10
    assert sum(r["тональность_отзыва"] == "negative" for r in rows) == 5
    path = tmp_path / "выборка.csv"
    write_sample(rows, path)
    sentences, gold = read_sample(path, list(TAXONOMY))
    assert len(sentences) == 10 and all(v is None for v in gold.values())


def test_run_model_on_fake_client():
    gold = {1: {"Ткань и материал": 1, "Качество пошива": -1}, 2: {}, 3: {"Размер и посадка": -1}}
    row = run_model(FakeClient([json.dumps(GOOD, ensure_ascii=False)]), SENTENCES, gold, TAXONOMY)
    assert row["f1"] == 1.0 and row["тональность_macro_f1"] == 1.0
    assert row["запросов"] == 1 and row["доля_ошибок_формата"] == 0.0
    assert "| fake | 1.0 |" in summary_table([row])
