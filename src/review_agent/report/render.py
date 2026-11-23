"""Шаг 12: вывод отчёта в Markdown и DOCX (FR-18, FR-20).

Оба формата строятся из одной структуры Отчёт, поэтому совпадают по содержанию.
DOCX открывается в Word и LibreOffice, оттуда же сохраняется в PDF.
"""
from __future__ import annotations

from pathlib import Path

from review_agent.report.build import СЛОЖНОСТЬ, Отчёт, процент


def _краткая_сводка(о: Отчёт) -> list[str]:
    строки = [f"Отзывов в анализе: {о.reviews}.",
              f"Найдено проблем: {len(о.problems)}, рекомендаций: {len(о.recommendations)}."]
    if о.problems:
        п = о.problems[0]
        строки.append(f"Самая частая проблема: «{п.title}» ({процент(п.coverage)} отзывов).")
    if о.recommendations:
        строки.append(f"Первое, что стоит сделать: {о.recommendations[0].action}")
    if о.strengths:
        строки.append("Сильные стороны: " + ", ".join(с.attribute for с in о.strengths) + ".")
    return строки


def _доля(доля: float | None) -> str:
    return "нет данных" if доля is None else процент(доля)


def в_markdown(о: Отчёт) -> str:
    с = [f"# Отчёт по отзывам: {о.product}", "", f"Дата: {о.created:%d.%m.%Y}", ""]
    с += [f"> {п}" for п in о.warnings] + ([""] if о.warnings else [])
    с += ["## Кратко", ""] + [f"- {x}" for x in _краткая_сводка(о)] + [""]

    с += ["## Рекомендации по приоритету", ""]
    if not о.recommendations:
        с += ["Рекомендаций, прошедших проверку, нет.", ""]
    for i, р in enumerate(о.recommendations, 1):
        с += [f"### {i}. {р.action}", "",
              f"- Проблема: {р.problem} ({р.attribute}), {процент(р.coverage)} отзывов",
              f"- Тип: {р.type}. Сложность: {СЛОЖНОСТЬ.get(р.effort, р.effort)}. "
              f"Приоритет: {р.priority:.4f} (доля негатива {_доля(р.negative_share)})"]
        if р.rationale:
            с.append(f"- Обоснование: {р.rationale}")
        if р.expected_effect:
            с.append(f"- Ожидаемый эффект: {р.expected_effect}")
        с += ["- Из отзывов:"] + [f"  - «{ц}»" for ц in р.quotes] + [""]

    с += ["## Проблемы", ""]
    if not о.problems:
        с += ["Повторяющихся проблем не найдено.", ""]
    for i, п in enumerate(о.problems, 1):
        без = "" if п.has_recommendation else " Рекомендация не получена или не прошла проверку."
        с += [f"### {i}. {п.title}", "",
              f"{п.attribute}: {п.reviews} отзывов ({процент(п.coverage)}).{без}", ""]
        с += [f"- «{ц}»" for ц in п.quotes] + [""]

    с += ["## Сильные стороны", ""]
    if not о.strengths:
        с += ["Не выявлены.", ""]
    for ст in о.strengths:
        с.append(f"- **{ст.attribute}**: {процент(ст.coverage)} отзывов, "
                 f"{процент(ст.positive_share)} упоминаний положительные")
        if ст.card_tip:
            с.append(f"  - Для карточки товара: {ст.card_tip}")
            с += [f"  - «{ц}»" for ц in ст.quotes]
    с.append("")

    if о.characteristics:
        с += ["## Характеристики", "",
              "| Характеристика | Упоминаний | Охват | Позитив | Негатив | Доля негатива | Средняя оценка |",
              "|---|---|---|---|---|---|---|"]
        for х in sorted(о.characteristics, key=lambda х: -х.mentions):
            оценка = "" if х.mean_rating is None else f"{х.mean_rating:.2f}"
            с.append(f"| {х.attribute} | {х.mentions} | {процент(х.coverage)} | {х.positive} | "
                     f"{х.negative} | {процент(х.negative_share)} | {оценка} |")
        с.append("")

    if о.rejected:
        с += ["## Приложение: отклонённые рекомендации", "",
              "Эти рекомендации не вошли в отчёт. Список нужен, чтобы видеть, что система отбросила и почему.", ""]
        с += [f"- [{п}] {д} — {причина}" for п, д, причина in о.rejected] + [""]

    с += ["---", "Отчёт собран автоматически по отзывам покупателей. Решение о действиях принимает продавец."]
    return "\n".join(с).rstrip() + "\n"


def в_docx(о: Отчёт, путь: str | Path) -> Path:
    """Сохраняет отчёт в DOCX. Нужна библиотека python-docx."""
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Cm, Pt, RGBColor

    д = Document()
    стиль = д.styles["Normal"]
    стиль.font.name = "Times New Roman"
    стиль.font.size = Pt(12)

    заголовок = д.add_heading(f"Отчёт по отзывам: {о.product}", level=0)
    заголовок.alignment = WD_ALIGN_PARAGRAPH.CENTER
    д.add_paragraph(f"Дата: {о.created:%d.%m.%Y}").alignment = WD_ALIGN_PARAGRAPH.CENTER
    for п in о.warnings:
        run = д.add_paragraph().add_run(п)
        run.italic = True
        run.font.color.rgb = RGBColor(0x9C, 0x57, 0x00)

    д.add_heading("Кратко", level=1)
    for x in _краткая_сводка(о):
        д.add_paragraph(x, style="List Bullet")

    д.add_heading("Рекомендации по приоритету", level=1)
    if not о.recommendations:
        д.add_paragraph("Рекомендаций, прошедших проверку, нет.")
    for i, р in enumerate(о.recommendations, 1):
        д.add_heading(f"{i}. {р.action}", level=2)
        поля = [(подпись, значение) for подпись, значение in (
            ("Проблема: ", f"{р.problem} ({р.attribute}), {процент(р.coverage)} отзывов"),
            ("Тип: ", f"{р.type}. Сложность: {СЛОЖНОСТЬ.get(р.effort, р.effort)}. "
                      f"Приоритет: {р.priority:.4f} (доля негатива {_доля(р.negative_share)})"),
            ("Обоснование: ", р.rationale),
            ("Ожидаемый эффект: ", р.expected_effect)) if значение]
        абзац = д.add_paragraph()
        for номер, (подпись, значение) in enumerate(поля):
            абзац.add_run(подпись).bold = True
            абзац.add_run(значение)
            if номер < len(поля) - 1:
                абзац.add_run().add_break()
        for ц in р.quotes:
            д.add_paragraph(f"«{ц}»", style="List Bullet").runs[0].italic = True

    д.add_heading("Проблемы", level=1)
    if not о.problems:
        д.add_paragraph("Повторяющихся проблем не найдено.")
    for i, п in enumerate(о.problems, 1):
        д.add_heading(f"{i}. {п.title}", level=2)
        без = "" if п.has_recommendation else " Рекомендация не получена или не прошла проверку."
        д.add_paragraph(f"{п.attribute}: {п.reviews} отзывов ({процент(п.coverage)}).{без}")
        for ц in п.quotes:
            д.add_paragraph(f"«{ц}»", style="List Bullet").runs[0].italic = True

    д.add_heading("Сильные стороны", level=1)
    if not о.strengths:
        д.add_paragraph("Не выявлены.")
    for ст in о.strengths:
        абзац = д.add_paragraph(style="List Bullet")
        абзац.add_run(ст.attribute).bold = True
        абзац.add_run(f": {процент(ст.coverage)} отзывов, {процент(ст.positive_share)} упоминаний положительные")
        if ст.card_tip:
            д.add_paragraph(f"Для карточки товара: {ст.card_tip}")

    if о.characteristics:
        д.add_heading("Характеристики", level=1)
        шапка = ["Характеристика", "Упом.", "Охват", "Позит.", "Негат.", "Доля негатива", "Ср. оценка"]
        таблица = д.add_table(rows=1, cols=len(шапка))
        таблица.style = "Table Grid"
        for ячейка, текст in zip(таблица.rows[0].cells, шапка):
            ячейка.text = текст
            ячейка.paragraphs[0].runs[0].bold = True
        for х in sorted(о.characteristics, key=lambda х: -х.mentions):
            строка = таблица.add_row().cells
            for ячейка, текст in zip(строка, [х.attribute, str(х.mentions), процент(х.coverage), str(х.positive),
                                              str(х.negative), процент(х.negative_share),
                                              "" if х.mean_rating is None else f"{х.mean_rating:.2f}"]):
                ячейка.text = текст
        # первая колонка шире: в ней названия характеристик, в остальных короткие числа
        таблица.autofit = False
        for номер, ширина in enumerate([Cm(4.6)] + [Cm(2.0)] * (len(шапка) - 1)):
            таблица.columns[номер].width = ширина      # сетка таблицы (её читает LibreOffice)
            for строка in таблица.rows:
                строка.cells[номер].width = ширина     # ширина ячеек (её читает Word)

    if о.rejected:
        д.add_heading("Приложение: отклонённые рекомендации", level=1)
        for п, действие, причина in о.rejected:
            д.add_paragraph(f"[{п}] {действие} — {причина}", style="List Bullet")

    д.add_paragraph().add_run("Отчёт собран автоматически по отзывам покупателей. "
                              "Решение о действиях принимает продавец.").italic = True
    путь = Path(путь)
    путь.parent.mkdir(parents=True, exist_ok=True)
    д.save(путь)
    return путь
