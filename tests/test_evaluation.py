import csv

import pytest

from review_agent.evaluation.annotation import load_annotation, parse_sentiment
from review_agent.evaluation.metrics import aspect_prf, cohen_kappa, evaluate, sentiment_scores, strict_prf

GOLD = {
    "s1": {"size": -1, "fabric": 1},
    "s2": {"look": 1},
    "s3": {},
}
PRED = {
    "s1": {"size": -1, "fabric": -1},
    "s2": {"look": 1, "match": 0},
    "s3": {},
    "лишнее": {"size": 1},
}


def test_aspect_prf():
    m = aspect_prf(GOLD, PRED)
    assert (m.tp, m.fp, m.fn) == (3, 1, 0)
    assert m.precision == pytest.approx(0.75)
    assert m.recall == 1.0


def test_strict_prf():
    m = strict_prf(GOLD, PRED)
    assert (m.tp, m.fp, m.fn) == (2, 2, 1)


def test_sentiment_scores():
    s = sentiment_scores(GOLD, PRED)
    assert s["pairs"] == 3
    assert s["accuracy"] == pytest.approx(2 / 3)


def test_kappa_perfect_and_partial():
    attrs = ["size", "fabric", "look", "match"]
    assert cohen_kappa(GOLD, GOLD, attrs) == pytest.approx(1.0)
    k = cohen_kappa(GOLD, PRED, attrs)
    assert 0 < k < 1


def test_evaluate_reports_missing():
    r = evaluate(GOLD, {"s1": {"size": -1}})
    assert r["missing_predictions"] == 2
    assert r["per_attribute"]["size"]["f1"] == 1.0


def test_parse_sentiment():
    assert parse_sentiment(" -1 ") == -1
    assert parse_sentiment("−1") == -1
    assert parse_sentiment(1.0) == 1
    assert parse_sentiment("") is None
    with pytest.raises(ValueError):
        parse_sentiment("2")


def test_load_annotation_csv(tmp_path):
    p = tmp_path / "a.csv"
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "sentence", "size", "fabric", "overall"])
        w.writerow(["x1", "Маломерит", "-1", "", ""])
        w.writerow(["x2", "Заказывала маме", "", "", ""])
    assert load_annotation(p) == {"x1": {"size": -1}, "x2": {}}


def test_load_real_xlsx_template():
    labels = load_annotation("data/annotation/mini.xlsx")
    assert len(labels) == 100
    assert all(v == {} for v in labels.values())
