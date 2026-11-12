"""Шаг 11 конвейера: проверка опоры на отзывы без LLM (FR-17).

Главная защита от выдуманных выводов: в отчёт попадает только то,
что можно подтвердить текстом настоящих отзывов.
"""
from review_agent.models import Issue, Recommendation, Sentence
from review_agent.pipeline.extract import _norm

MIN_SENTENCES = 3  # проблема должна опираться минимум на столько предложений


def check_grounding(
    recommendations: list[Recommendation], issues: list[Issue], sentences: list[Sentence],
    min_sentences: int = MIN_SENTENCES,
) -> tuple[list[Recommendation], list[tuple[Recommendation, str]]]:
    """Делит рекомендации на принятые и отклонённые (с причиной отказа)."""
    text_by_id = {s.id: _norm(s.text) for s in sentences}
    issue_by_id = {i.id: i for i in issues}
    accepted, rejected = [], []
    for item in recommendations:
        reason = _reason(item, issue_by_id.get(item.issue_id), text_by_id, min_sentences)
        item.grounded = reason is None
        if reason is None:
            accepted.append(item)
        else:
            rejected.append((item, reason))
    return accepted, rejected


def _reason(item: Recommendation, issue: Issue | None, text_by_id: dict[int, str], min_sentences: int) -> str | None:
    if issue is None:
        return f"рекомендация ссылается на несуществующую проблему {item.issue_id}"
    existing = [i for i in issue.sentence_ids if i in text_by_id]
    if len(existing) < min_sentences:
        return f"проблема опирается на {len(existing)} предложений, нужно не меньше {min_sentences}"
    if not item.quotes:
        return "у рекомендации нет цитат из отзывов"
    for sentence_id, quote in item.quotes:
        if sentence_id not in issue.sentence_ids:
            return f"цитата взята из предложения {sentence_id}, которое не относится к проблеме"
        if not _norm(quote) or _norm(quote) not in text_by_id.get(sentence_id, ""):
            return f"цитаты «{quote}» нет в предложении {sentence_id}"
    return None
