"""Метрики качества извлечения аспектов и тональности.

Все функции принимают разметку в виде {id предложения: {атрибут: тональность}}.
Оценка идёт только по предложениям, которые есть в эталоне.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

Labels = dict[str, dict[str, int]]
SENTIMENTS = (-1, 0, 1)


@dataclass
class PRF:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def as_dict(self) -> dict[str, float]:
        return {"precision": self.precision, "recall": self.recall, "f1": self.f1, "support": self.tp + self.fn}


def _prf(gold: set, pred: set) -> PRF:
    return PRF(tp=len(gold & pred), fp=len(pred - gold), fn=len(gold - pred))


def aspect_prf(gold: Labels, pred: Labels) -> PRF:
    """Нашла ли модель нужные атрибуты в предложении (без учёта тональности)."""
    g = {(s, a) for s, m in gold.items() for a in m}
    p = {(s, a) for s, m in pred.items() if s in gold for a in m}
    return _prf(g, p)


def strict_prf(gold: Labels, pred: Labels) -> PRF:
    """Атрибут и тональность совпадают одновременно."""
    g = {(s, a, v) for s, m in gold.items() for a, v in m.items()}
    p = {(s, a, v) for s, m in pred.items() if s in gold for a, v in m.items()}
    return _prf(g, p)


def per_attribute_prf(gold: Labels, pred: Labels) -> dict[str, PRF]:
    attrs = sorted({a for m in gold.values() for a in m} | {a for s, m in pred.items() if s in gold for a in m})
    result = {}
    for attr in attrs:
        g = {s for s, m in gold.items() if attr in m}
        p = {s for s, m in pred.items() if s in gold and attr in m}
        result[attr] = _prf(g, p)
    return result


def sentiment_scores(gold: Labels, pred: Labels) -> dict[str, float]:
    """Accuracy и macro-F1 тональности на парах, найденных и человеком, и моделью."""
    pairs = [(v, pred[s][a]) for s, m in gold.items() for a, v in m.items() if a in pred.get(s, {})]
    if not pairs:
        return {"accuracy": 0.0, "macro_f1": 0.0, "pairs": 0}
    acc = sum(g == p for g, p in pairs) / len(pairs)
    f1s = []
    for c in SENTIMENTS:
        tp = sum(g == c and p == c for g, p in pairs)
        fp = sum(g != c and p == c for g, p in pairs)
        fn = sum(g == c and p != c for g, p in pairs)
        # Класс, которого нет ни в эталоне, ни в ответах, не учитываем
        if tp + fp + fn:
            f1s.append(PRF(tp, fp, fn).f1)
    return {"accuracy": acc, "macro_f1": sum(f1s) / len(f1s), "pairs": len(pairs)}


def cohen_kappa(a: Labels, b: Labels, attributes: list[str]) -> float:
    """Каппа Коэна между двумя разметками.

    Единица сравнения — ячейка (предложение, атрибут) со значениями
    «нет», −1, 0, 1. Берутся только предложения, размеченные обоими.
    """
    common = sorted(set(a) & set(b))
    pairs = [(a[s].get(attr, "нет"), b[s].get(attr, "нет")) for s in common for attr in attributes]
    if not pairs:
        raise ValueError("Нет общих предложений")
    n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum(ca[k] * cb[k] for k in ca) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def evaluate(gold: Labels, pred: Labels) -> dict:
    """Сводка всех метрик для отчёта и главы 3."""
    missing = sorted(set(gold) - set(pred))
    return {
        "sentences": len(gold),
        "missing_predictions": len(missing),
        "aspects": aspect_prf(gold, pred).as_dict(),
        "strict": strict_prf(gold, pred).as_dict(),
        "sentiment": sentiment_scores(gold, pred),
        "per_attribute": {k: v.as_dict() for k, v in per_attribute_prf(gold, pred).items()},
    }
