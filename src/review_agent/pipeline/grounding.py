"""Шаг 11. Проверка опоры на отзывы без LLM (FR-17).

Главная защита от выдуманных выводов: рекомендация должна ссылаться на
существующую проблему, проблема — минимум на min_support реальных
предложений, а все цитаты должны находиться в исходном тексте.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from review_agent.config import Thresholds
from review_agent.models import Issue, Mention, Recommendation, Sentence
from review_agent.pipeline.extract import quote_in_sentence


@dataclass
class GroundingReport:
    accepted: list[Recommendation] = field(default_factory=list)
    rejected: list[tuple[Recommendation, list[str]]] = field(default_factory=list)
    bad_issues: dict[str, list[str]] = field(default_factory=dict)


def check_issue(
    issue: Issue,
    mentions: dict[str, Mention],
    sentences: dict[str, Sentence],
    min_support: int = Thresholds.min_support,
) -> list[str]:
    problems = []
    support: set[str] = set()
    for mid in issue.mention_ids:
        m = mentions.get(mid)
        if m is None:
            problems.append(f"упоминание {mid} не найдено")
            continue
        s = sentences.get(m.sentence_id)
        if s is None or not quote_in_sentence(m.quote, s.text):
            problems.append(f"цитата упоминания {mid} не совпадает с предложением")
            continue
        support.add(s.id)
    if len(support) < min_support:
        problems.append(f"опора {len(support)} предложений, нужно не меньше {min_support}")
    return problems


def check_recommendation(
    rec: Recommendation,
    issues: dict[str, Issue],
    bad_issues: dict[str, list[str]],
    mentions: dict[str, Mention],
    sentences: dict[str, Sentence],
) -> list[str]:
    problems = []
    if rec.issue_id is not None:
        if rec.issue_id not in issues:
            return [f"проблема {rec.issue_id} не существует"]
        if rec.issue_id in bad_issues:
            problems.append(f"проблема {rec.issue_id} не прошла проверку")
        # Цитаты рекомендации ищем в предложениях её проблемы
        texts = [sentences[mentions[mid].sentence_id].text for mid in issues[rec.issue_id].mention_ids if mid in mentions]
    else:
        texts = [s.text for s in sentences.values()]
    if not rec.quotes:
        problems.append("нет ни одной цитаты из отзывов")
    for q in rec.quotes:
        if not any(quote_in_sentence(q, t) for t in texts):
            problems.append(f"цитата «{q}» не найдена в отзывах")
    return problems


def apply_grounding(
    recommendations: list[Recommendation],
    issues: list[Issue],
    mentions: list[Mention],
    sentences: list[Sentence],
    min_support: int = Thresholds.min_support,
) -> GroundingReport:
    issue_by_id = {i.id: i for i in issues}
    mention_by_id = {m.id: m for m in mentions}
    sentence_by_id = {s.id: s for s in sentences}
    report = GroundingReport()
    for issue in issues:
        problems = check_issue(issue, mention_by_id, sentence_by_id, min_support)
        if problems:
            report.bad_issues[issue.id] = problems
    for rec in recommendations:
        problems = check_recommendation(rec, issue_by_id, report.bad_issues, mention_by_id, sentence_by_id)
        if problems:
            report.rejected.append((rec.model_copy(update={"grounded": False}), problems))
        else:
            report.accepted.append(rec.model_copy(update={"grounded": True}))
    return report
