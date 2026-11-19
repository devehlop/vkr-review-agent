from review_agent.evaluation.baseline import predict, sentence_sentiment


def test_baseline_attributes_and_sentiment():
    p = predict({
        "a": "Ткань тонкая, швы кривые",
        "b": "Отличное платье, рекомендую",
        "c": "Заказывала маме",
        "d": "Размер не подошел",
    })
    assert p["a"] == {"fabric": -1, "sewing": -1}
    assert p["b"] == {"overall": 1}
    assert p["c"] == {}
    assert p["d"] == {"size": -1}


def test_negation():
    assert sentence_sentiment("не понравилось") == -1
    assert sentence_sentiment("не плохо") == 1
