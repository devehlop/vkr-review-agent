"""Шаг 12 конвейера: отчёт для продавца в формате Markdown (FR-18).

Текст отчёта собирается кодом из проверенных данных, без обращения к модели.
"""
from review_agent.pipeline.compare import BETTER, WORSE
from review_agent.pipeline.orchestrator import PipelineResult
from review_agent.pipeline.recommend import percent

EFFORT = {1: "низкая", 2: "средняя", 3: "высокая"}


def build_report(result: PipelineResult, min_mentions: int = 5, examples: int = 3) -> str:
    own = result.own_product
    text_by_id = {s.id: s.text for s in result.sentences}
    own_reviews = sum(r.product == own for r in result.reviews)
    competitors = sorted({r.product for r in result.reviews if r.product != own})
    own_stats = [s for s in result.stats if s.product == own and not s.feature]
    issue_by_id = {i.id: i for i in result.issues}

    lines = [f"# Отчёт по отзывам: {own}", ""]
    lines += [f"> ⚠️ {w}" for w in result.warnings] + ([""] if result.warnings else [])

    lines += ["## Кратко", "",
              f"- Отзывов на товар: {own_reviews}" + (f", конкуренты: {', '.join(competitors)}" if competitors else ""),
              f"- Найдено проблем: {len(result.issues)}, рекомендаций: {len(result.recommendations)}"]
    if result.issues:
        top = result.issues[0]
        lines.append(f"- Самая частая проблема: «{top.title}» ({percent(top.coverage)} отзывов)")
    if result.recommendations:
        lines.append(f"- Первое, что стоит сделать: {result.recommendations[0].action}")
    lines.append("")

    strong = sorted((s for s in own_stats if s.mentions >= min_mentions and s.positive / s.mentions >= 0.7),
                    key=lambda s: -s.coverage)
    lines += ["## Сильные стороны", ""]
    lines += [f"- **{s.attribute}**: упоминается в {percent(s.coverage)} отзывов, "
              f"{percent(s.positive / s.mentions)} упоминаний положительные" for s in strong] or ["Не выявлены."]
    lines.append("")

    lines += ["## Проблемы", ""]
    if not result.issues:
        lines += ["Повторяющихся проблем не найдено.", ""]
    for issue in result.issues:
        lines += [f"### {issue.id}. {issue.title}", "",
                  f"Характеристика: {issue.attribute}. Встречается в {percent(issue.coverage)} отзывов, "
                  f"упоминаний: {len(issue.sentence_ids)}."
                  + (" У конкурентов встречается заметно реже." if issue.rarer_at_competitors else ""), "",
                  "Примеры из отзывов:"]
        lines += [f"- «{text_by_id[i]}»" for i in issue.sentence_ids[:examples] if i in text_by_id] + [""]

    compared = [c for c in result.comparisons if c.verdict in (WORSE, BETTER)]
    if competitors:
        lines += ["## Сравнение с конкурентами", ""]
        if compared:
            lines += ["| Характеристика | Наш товар, доля негатива | " + " | ".join(competitors) + " | Вывод |",
                      "|---|---|" + "---|" * (len(competitors) + 1)]
            for c in compared:
                cells = [percent(c.competitors[name][0]) if name in c.competitors else "нет данных" for name in competitors]
                lines.append(f"| {c.attribute} | {percent(c.own_neg_share)} | " + " | ".join(cells) + f" | {c.verdict} |")
        else:
            lines.append("Заметных различий не найдено или данных недостаточно.")
        lines.append("")

    lines += ["## Рекомендации по приоритету", ""]
    if not result.recommendations:
        lines += ["Рекомендаций, прошедших проверку, нет.", ""]
    for number, rec in enumerate(result.recommendations, start=1):
        issue = issue_by_id[rec.issue_id]
        lines += [f"### {number}. {rec.action}", "",
                  f"- Проблема: {issue.title} ({percent(issue.coverage)} отзывов)",
                  f"- Тип: {rec.type}. Сложность: {EFFORT[rec.effort]}. Приоритет: {rec.priority}",
                  f"- Обоснование: {rec.rationale}"]
        if rec.expected_effect:
            lines.append(f"- Ожидаемый эффект: {rec.expected_effect}")
        lines += ["- Отзывы:"] + [f"  - «{text_by_id.get(sid, quote)}»" for sid, quote in rec.quotes] + [""]
    return "\n".join(lines).rstrip() + "\n"
