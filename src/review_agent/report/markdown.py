"""Шаг 12. Отчёт для продавца в Markdown (FR-18)."""

from __future__ import annotations

from review_agent.pipeline.orchestrator import AnalysisResult


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.0f} %"


def render_markdown(r: AnalysisResult) -> str:
    own = [p for p in r.products if p.is_own]
    comps = [p for p in r.products if not p.is_own]
    n_own = sum(rv.product_id in r.own_ids for rv in r.reviews)
    lines = [f"# Отчёт по отзывам: {', '.join(p.name for p in own)}", ""]

    for w in r.warnings:
        lines.append(f"> ⚠ {w}")
    if r.warnings:
        lines.append("")

    # Резюме
    lines += ["## Кратко", ""]
    lines.append(f"- Проанализировано отзывов на свой товар: {n_own}" + (f", на конкурентов: {len(r.reviews) - n_own} ({', '.join(p.name for p in comps)})" if comps else ""))
    lines.append(f"- Выявлено проблем: {len(r.issues)}, рекомендаций: {len(r.recommendations)}")
    if r.issues:
        top = max(r.issues, key=lambda i: i.coverage)
        lines.append(f"- Самая частая проблема: «{top.title}» — {_pct(top.coverage)} отзывов")
    worse = [c.attribute for c in r.comparison if c.verdict == "хуже"]
    if worse:
        lines.append(f"- Отставание от конкурентов: {', '.join(worse)}")
    lines.append("")

    # Рекомендации
    issue_by_id = {i.id: i for i in r.issues}
    lines += ["## Рекомендации по приоритету", ""]
    if not r.recommendations:
        lines.append("Рекомендаций, прошедших проверку, нет.")
    for k, rec in enumerate(r.recommendations, start=1):
        issue = issue_by_id.get(rec.issue_id or "")
        target = f"проблема «{issue.title}»" if issue else "сильная сторона"
        lines.append(f"### {k}. {rec.action}")
        lines.append(f"*Тип:* {rec.type.value} · *сложность:* {rec.effort} из 3 · *приоритет:* {rec.priority:.3f} · *{target}*")
        lines.append("")
        lines.append(rec.rationale)
        if rec.expected_effect:
            lines.append(f"\n*Ожидаемый эффект:* {rec.expected_effect}")
        lines.append("")
        lines += [f"> «{q}»" for q in rec.quotes]
        lines.append("")

    # Проблемы
    mention_by_id = {m.id: m for m in r.mentions}
    lines += ["## Проблемы", ""]
    if not r.issues:
        lines.append("Повторяющихся проблем (от 3 упоминаний) не найдено.")
    for i in sorted(r.issues, key=lambda x: -x.coverage):
        quotes = list(dict.fromkeys(mention_by_id[m].quote for m in i.mention_ids if m in mention_by_id))[:5]
        lines.append(f"- **{i.title}** ({i.attribute}): {_pct(i.coverage)} отзывов, {len(i.mention_ids)} упоминаний. "
                     + "; ".join(f"«{q}»" for q in quotes))
    lines.append("")

    # Сильные стороны и атрибуты
    lines += ["## Атрибуты товара", "", "| Атрибут | Охват | Негатив | Позитив | Средняя оценка |", "|---|---|---|---|---|"]
    for s in sorted((s for s in r.stats if s.product_id in r.own_ids), key=lambda s: -s.coverage):
        avg = "—" if s.avg_rating is None else f"{s.avg_rating:.1f}"
        lines.append(f"| {s.attribute} | {_pct(s.coverage)} | {_pct(s.neg_share)} | {_pct(s.pos_share)} | {avg} |")
    lines.append("")

    # Сравнение
    if comps:
        lines += ["## Сравнение с конкурентами", "", "Доля отзывов с жалобами на атрибут.", "",
                  "| Атрибут | Свой товар | Конкуренты | Итог |", "|---|---|---|---|"]
        for c in r.comparison:
            lines.append(f"| {c.attribute} | {_pct(c.own_rate)} | {_pct(c.competitor_rate)} | {c.verdict} |")
        lines.append("")

    # Служебное
    tot = {k: sum(v[k] for v in r.llm_summary.values()) for k in ("calls", "tokens_in", "tokens_out", "cost")} if r.llm_summary else {}
    lines += ["## Служебная информация", ""]
    if tot:
        lines.append(f"- Запросов к модели: {tot['calls']:.0f}, токенов: {tot['tokens_in'] + tot['tokens_out']:.0f}, стоимость: {tot['cost']:.2f} ₽")
    lines.append(f"- Удалено дубликатов и пустых отзывов: {r.removed_duplicates}")
    skipped = sum(len(v) for v in r.skipped_rows.values())
    if skipped:
        lines.append(f"- Пропущено строк файла: {skipped}")
    if r.failed_sentence_ids:
        lines.append(f"- Предложений без ответа модели: {len(r.failed_sentence_ids)}")
    lines.append(f"- Отклонено критиком: {len(r.rejected_by_critic)}, отклонено проверкой опоры на отзывы: {len(r.ungrounded)}")
    lines.append(f"- Справочник аспектов: {r.taxonomy.version}")
    return "\n".join(lines) + "\n"
