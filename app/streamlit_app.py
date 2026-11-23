"""Веб-интерфейс для продавца (FR-19).

Запуск из корня репозитория:
    streamlit run app/streamlit_app.py

Два режима:
- «Новый анализ» запускает всю цепочку шагов 2–12. Нужен ключ GigaChat
  в переменной GIGACHAT_CREDENTIALS (и сертификат в GIGACHAT_CA_BUNDLE), как для скриптов.
- «Открыть результаты» строит отчёт по сохранённым файлам без модели.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from review_agent import app_support as поддержка  # noqa: E402
from review_agent.ingest.loader import ЗагрузкаОшибка, загрузить  # noqa: E402
from review_agent.llm.gateway import LLMGateway, LLMGatewayError  # noqa: E402
from review_agent.pipeline.run_all import ИтогАнализа, запустить_анализ, открыть_сохранённое, сохранить  # noqa: E402
from review_agent.pipeline.taxonomy import Справочник  # noqa: E402
from review_agent.report.build import СЛОЖНОСТЬ, Отчёт, прочитать_агрегат, процент  # noqa: E402
from review_agent.report.render import в_docx, в_markdown  # noqa: E402

st.set_page_config(page_title="Анализ отзывов", page_icon="🧵", layout="wide")
ПРИМЕР = поддержка.КОРЕНЬ / "data" / "synthetic_reviews.csv"


# ---------- ввод ----------
def выбрать_отзывы(ключ: str) -> tuple[Path, str] | None:
    """Файл с отзывами и название товара. Возвращает (путь, товар) или None."""
    левая, правая = st.columns([2, 1])
    with левая:
        файл = st.file_uploader("Файл с отзывами на свой товар (CSV или XLSX)", type=["csv", "xlsx"],
                                key=f"{ключ}_file")
        пример = файл is None and ПРИМЕР.is_file() and st.checkbox(
            "Взять пример: 54 выдуманных отзыва на куртку (data/synthetic_reviews.csv)", key=f"{ключ}_example")
    with правая:
        товар = st.text_input("Название товара", value="Куртка" if пример else "", key=f"{ключ}_product")
    if файл is not None:
        путь = поддержка.сохранить_загруженный(файл.getvalue(), файл.name)
    elif пример:
        путь = ПРИМЕР
    else:
        return None
    if not товар.strip():
        st.warning("Укажите название товара.")
        return None
    return путь, товар.strip()


def прочитать_отзывы(путь: Path, товар: str):
    try:
        отзывы = загрузить(путь, product=товар)
    except ЗагрузкаОшибка as ошибка:
        st.error(f"Не удалось прочитать файл: {ошибка}")
        return None
    st.caption(f"Прочитано отзывов: {len(отзывы)}. Колонки определены автоматически; "
               "имена покупателей и другие лишние колонки не читаются.")
    return отзывы


# ---------- вывод ----------
def показать_цитаты(цитаты: list[str]) -> None:
    for ц in цитаты:
        st.markdown(f"> {ц}")


def раздел_рекомендаций(о: Отчёт) -> None:
    if not о.recommendations:
        st.info("Рекомендаций, прошедших проверку, нет.")
    for i, р in enumerate(о.recommendations, 1):
        with st.expander(f"{i}. {р.action}", expanded=i == 1):
            st.markdown(f"**Проблема:** {р.problem} ({р.attribute}), {процент(р.coverage)} отзывов")
            доля = "нет данных" if р.negative_share is None else процент(р.negative_share)
            st.caption(f"Тип: {р.type} · Сложность: {СЛОЖНОСТЬ.get(р.effort, р.effort)} · "
                       f"Приоритет: {р.priority:.4f} · Доля негатива: {доля}")
            if р.rationale:
                st.markdown(f"**Обоснование:** {р.rationale}")
            if р.expected_effect:
                st.markdown(f"**Ожидаемый эффект:** {р.expected_effect}")
            st.markdown("**Из отзывов:**")
            показать_цитаты(р.quotes)


def раздел_проблем(о: Отчёт) -> None:
    if not о.problems:
        st.info("Повторяющихся проблем не найдено.")
    for i, п in enumerate(о.problems, 1):
        st.markdown(f"#### {i}. {п.title}")
        st.caption(f"{п.attribute} · {п.reviews} отзывов ({процент(п.coverage)})"
                   + ("" if п.has_recommendation else " · рекомендация не получена или не прошла проверку"))
        показать_цитаты(п.quotes)


def раздел_сильных(о: Отчёт) -> None:
    if not о.strengths:
        st.info("Сильные стороны не выявлены.")
    for с in о.strengths:
        st.markdown(f"**{с.attribute}**: {процент(с.coverage)} отзывов, "
                    f"{процент(с.positive_share)} упоминаний положительные")
        if с.card_tip:
            st.markdown(f"Для карточки товара: {с.card_tip}")
            показать_цитаты(с.quotes)


def раздел_характеристик(о: Отчёт) -> None:
    if not о.characteristics:
        st.info("Нет данных о характеристиках.")
        return
    строки = sorted(о.characteristics, key=lambda х: -х.mentions)
    таблица = pd.DataFrame({
        "Характеристика": [х.attribute for х in строки],
        "Упоминаний": [х.mentions for х in строки],
        "Охват, %": [round(х.coverage * 100) for х in строки],
        "Негатив": [х.negative for х in строки],
        "Нейтрально": [х.neutral for х in строки],
        "Позитив": [х.positive for х in строки],
        "Доля негатива, %": [round(х.negative_share * 100) for х in строки],
        "Средняя оценка": [х.mean_rating for х in строки],
    })
    st.caption("Охват: доля отзывов, где характеристика упомянута. Доля негатива: среди её упоминаний.")
    st.dataframe(таблица, hide_index=True, width="stretch")
    st.bar_chart(таблица.set_index("Характеристика")[["Негатив", "Нейтрально", "Позитив"]],
                 color=["#c0392b", "#95a5a6", "#27ae60"], horizontal=True)


def раздел_журнала(о: Отчёт, итог: ИтогАнализа | None) -> None:
    st.markdown("**Отклонённые рекомендации** (в отчёт не вошли)")
    if о.rejected:
        st.dataframe(pd.DataFrame(о.rejected, columns=["Проблема", "Действие", "Причина"]),
                     hide_index=True, width="stretch")
    else:
        st.write("Нет.")
    if итог is None:
        st.caption("Расход модели показывается только для нового анализа.")
        return
    в = итог.stats["llm"]
    колонки = st.columns(4)
    колонки[0].metric("Вызовов модели", в["calls"])
    колонки[1].metric("Из кэша", в["cache_hits"])
    колонки[2].metric("Ошибок", в["errors"])
    колонки[3].metric("Токенов", f"{в['tokens_in']} + {в['tokens_out']}")
    п = итог.preprocess
    st.write(f"Загружено отзывов: {п['loaded']}, пустых: {п['empty_removed']}, дубликатов: "
             f"{п['duplicates_removed']}, в анализе: {п['kept']}, подозрительных: {п['suspicious']}, "
             f"предложений: {п['sentences']}.")
    for шаг, ошибка in итог.errors:
        st.caption(f"Ошибка [{шаг}]: {ошибка}")
    st.markdown("**Справочник характеристик**")
    st.dataframe(pd.DataFrame({"Характеристика": [а.name for а in итог.taxonomy.attributes],
                               "Признаки": [", ".join(а.features) for а in итог.taxonomy.attributes]}),
                 hide_index=True, width="stretch")


def показать(о: Отчёт, итог: ИтогАнализа | None = None, папка: Path | None = None) -> None:
    st.divider()
    st.header(f"Результат: {о.product}")
    for п in о.warnings:
        st.warning(п)
    колонки = st.columns(4)
    колонки[0].metric("Отзывов", о.reviews)
    колонки[1].metric("Проблем", len(о.problems))
    колонки[2].metric("Рекомендаций", len(о.recommendations))
    колонки[3].metric("Сильных сторон", len(о.strengths))
    if папка:
        st.caption(f"Файлы результатов сохранены в {папка}" if итог
                   else f"Результаты открыты из {папка}; шаги 10–11 пересчитаны.")

    разделы = {"Рекомендации": раздел_рекомендаций, "Проблемы": раздел_проблем,
               "Сильные стороны": раздел_сильных, "Характеристики": раздел_характеристик}
    выбор = st.segmented_control("Раздел", [*разделы, "Журнал"], default="Рекомендации", key="раздел",
                                 label_visibility="collapsed")
    if выбор == "Журнал":
        раздел_журнала(о, итог)
    else:
        разделы.get(выбор or "Рекомендации", раздел_рекомендаций)(о)

    with tempfile.TemporaryDirectory() as временная:
        docx = в_docx(о, Path(временная) / "report.docx").read_bytes()
    левая, правая = st.columns(2)
    левая.download_button("Скачать отчёт (DOCX)", docx, file_name="report.docx",
                          mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    правая.download_button("Скачать отчёт (Markdown)", в_markdown(о), file_name="report.md",
                           mime="text/markdown")


# ---------- режимы ----------
def новый_анализ() -> None:
    выбор = выбрать_отзывы("new")
    with st.sidebar:
        st.subheader("Настройки анализа")
        категория = st.text_input("Категория товара", value="женская одежда")
        справочники = sorted((поддержка.КОРЕНЬ / "data").glob("taxonomy*.json"))
        вариант = st.selectbox("Справочник характеристик", ["Составит модель", *[п.name for п in справочники]],
                               help="Готовый справочник экономит запрос к модели и делает прогоны сравнимыми.")
        с_критиком = st.checkbox("Проверять рекомендации критиком", value=True)
        эмбеддер = st.radio("Эмбеддинги для группировки жалоб", ["local", "gigachat"],
                            format_func={"local": "на компьютере (бесплатно)", "gigachat": "GigaChat (платно)"}.get)
        порог = st.slider("Порог кластеризации", 0.10, 0.60, 0.35, 0.05,
                          help="Меньше — проблемы мельче и их больше.")
    if not поддержка.ключ_задан():
        st.info("Для нового анализа задайте ключ GigaChat в терминале перед запуском интерфейса: "
                "`read -s \"GIGACHAT_CREDENTIALS?Ключ: \"; export GIGACHAT_CREDENTIALS` и "
                "`export GIGACHAT_CA_BUNDLE=путь/к/сертификату`. Без ключа можно открыть сохранённые результаты.")
    if выбор is None:
        return
    путь, товар = выбор
    отзывы = прочитать_отзывы(путь, товар)
    if not отзывы or not st.button("Запустить анализ", type="primary"):
        return
    справочник = None if вариант == "Составит модель" else Справочник.загрузить(поддержка.КОРЕНЬ / "data" / вариант)
    try:
        клиент = поддержка.создать_клиента()
        кэш = поддержка.КОРЕНЬ / ".cache"
        кэш.mkdir(exist_ok=True)
        шлюз = LLMGateway(клиент, db_path=str(кэш / "llm.sqlite"))
        with st.status("Идёт анализ…", expanded=True) as статус:
            итог = запустить_анализ(шлюз, поддержка.создать_эмбеддер(эмбеддер, клиент), отзывы, товар=товар,
                                    категория=категория, справочник=справочник, с_критиком=с_критиком,
                                    порог_кластеризации=порог, шаг=st.write)
            папка = сохранить(итог, поддержка.папка_нового_прогона(товар))
            статус.update(label="Анализ завершён", state="complete", expanded=False)
    except (LLMGatewayError, RuntimeError, ValueError) as ошибка:
        # нет ключа, модель недоступна или не ответила по схеме, нет отзывов на товар
        st.error(f"Анализ не выполнен: {ошибка}")
        return
    st.session_state["результат"] = (итог.report, итог, папка)


def открыть_результаты() -> None:
    папки = поддержка.папки_с_результатами()
    if not папки:
        st.info("Сохранённых результатов нет: нужны issues.json и aggregate.csv в папке runs/ или data/.")
        return
    папка = st.selectbox("Папка с результатами", папки,
                         format_func=lambda п: str(п.relative_to(поддержка.КОРЕНЬ)))
    рекомендации = папка / "recommendations.json"
    st.caption("Есть рекомендации (шаги 8–9)." if рекомендации.is_file()
               else "Файла recommendations.json нет: в отчёте будут только проблемы и характеристики.")
    выбор = выбрать_отзывы("open")
    if выбор is None:
        return
    путь, товар = выбор
    отзывы = прочитать_отзывы(путь, товар)
    if not отзывы or not st.button("Открыть", type="primary"):
        return
    try:
        отчёт = открыть_сохранённое(
            отзывы, товар=товар,
            проблемы=json.loads((папка / "issues.json").read_text(encoding="utf-8")),
            агрегат=прочитать_агрегат(папка / "aggregate.csv", товар),
            рекомендации=json.loads(рекомендации.read_text(encoding="utf-8")) if рекомендации.is_file() else None)
    except (ValueError, KeyError, TypeError) as ошибка:
        st.error(f"Не удалось открыть результаты: {ошибка}")
        return
    st.session_state["результат"] = (отчёт, None, папка)


st.title("Анализ отзывов покупателей")
st.caption("Агент находит повторяющиеся проблемы товара по отзывам и предлагает, что изменить в первую очередь. "
           "Каждая рекомендация опирается на цитаты из отзывов.")
with st.sidebar:
    режим = st.radio("Режим", ["Новый анализ", "Открыть результаты"])
if st.session_state.get("режим") != режим:
    st.session_state["режим"] = режим
    st.session_state.pop("результат", None)
(новый_анализ if режим == "Новый анализ" else открыть_результаты)()
if "результат" in st.session_state:
    показать(*st.session_state["результат"])
