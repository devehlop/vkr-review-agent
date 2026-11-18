"""Метрики качества извлечения аспектов и тональности (раздел 7.2 ТЗ)."""

CLASSES = (-1, 0, 1)


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def aspect_metrics(gold: dict[int, dict[str, int]], predicted: dict[int, dict[str, int]]) -> dict:
    """Сравнивает эталон и ответ модели.

    Единица сравнения: пара «предложение, атрибут».
    Precision: какая доля найденных моделью пар есть в эталоне.
    Recall: какая доля пар эталона найдена моделью.
    Тональность оценивается только на парах, найденных верно.
    """
    gold_pairs = {(i, a) for i, labels in gold.items() for a in labels}
    pred_pairs = {(i, a) for i, labels in predicted.items() if i in gold for a in labels}
    matched = gold_pairs & pred_pairs
    precision = len(matched) / len(pred_pairs) if pred_pairs else 0.0
    recall = len(matched) / len(gold_pairs) if gold_pairs else 0.0

    correct = sum(gold[i][a] == predicted[i][a] for i, a in matched)
    f1_by_class = []
    for cls in CLASSES:
        tp = sum(gold[i][a] == cls and predicted[i][a] == cls for i, a in matched)
        in_gold = sum(gold[i][a] == cls for i, a in matched)
        in_pred = sum(predicted[i][a] == cls for i, a in matched)
        if in_gold or in_pred:  # класс, которого нет ни в эталоне, ни в ответе, в среднее не входит
            f1_by_class.append(_f1(tp / in_pred if in_pred else 0.0, tp / in_gold if in_gold else 0.0))

    # Предложения без упоминаний: верно ли модель оставила их пустыми
    empty_gold = [i for i, labels in gold.items() if not labels]
    empty_ok = sum(not predicted.get(i) for i in empty_gold)
    return {
        "пар_в_эталоне": len(gold_pairs),
        "пар_у_модели": len(pred_pairs),
        "совпало": len(matched),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(_f1(precision, recall), 3),
        "тональность_accuracy": round(correct / len(matched), 3) if matched else 0.0,
        "тональность_macro_f1": round(sum(f1_by_class) / len(f1_by_class), 3) if f1_by_class else 0.0,
        "пустых_в_эталоне": len(empty_gold),
        "пустых_верно": empty_ok,
    }
