"""Веб-интерфейс для продавца (FR-19).

Запуск из корня репозитория:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from review_agent.config import Settings  # noqa: E402
from review_agent.ingest.loader import ColumnMapping, guess_mapping, read_table  # noqa: E402
from review_agent.llm import LLMGateway, make_client  # noqa: E402
from review_agent.llm.demo import demo_client  # noqa: E402
from review_agent.models import Product  # noqa: E402
from review_agent.pipeline.embeddings import SentenceTransformerEmbedder, TfidfEmbedder  # noqa: E402
from review_agent.pipeline.orchestrator import AnalysisResult, ProductInput, run_analysis  # noqa: E402
from review_agent.report.markdown import render_markdown  # noqa: E402
from review_agent.storage.db import Storage  # noqa: E402
from review_agent.taxonomy import load_taxonomy  # noqa: E402

FIELDS = {"text": "Текст / комментарий", "pros": "Достоинства", "cons": "Недостатки", "rating": "Оценка",
          "created": "Дата", "variant": "Вариант (размер, цвет)", "id": "Номер отзыва"}
NONE = "— нет —"
DEMO = [("Свой товар (демо)", ROOT / "data/samples/demo_own.csv", True),
        ("Конкурент (демо)", ROOT / "data/samples/demo_competitor.csv", False)]

st.set_page_config(page_title="Анализ отзывов", page_icon="🧵", layout="wide")
st.title("Анализ отзывов покупателей")
st.caption("Загрузите выгрузку отзывов своего товара и, по желанию, конкурентов. Агент найдёт проблемы и предложит, что исправить.")

# --- Настройки -----------------------------------------------------------
with st.sidebar:
    st.header("Настройки")
    provider = st.selectbox("Языковая модель", ["demo", "gigachat", "yandex", "ollama"],
                            format_func=lambda p: {"demo": "Демо (без модели)", "gigachat": "GigaChat",
                                                   "yandex": "YandexGPT", "ollama": "Локальная (Ollama)"}[p])
    model = st.text_input("Модель", value="", placeholder="по умолчанию")
    taxonomy_choice = st.selectbox("Справочник аспектов", ["clothing_v1", "auto"],
                                   format_func=lambda t: "Одежда (встроенный)" if t == "clothing_v1" else "Построить по отзывам")
    embedder_choice = st.selectbox("Поиск похожих жалоб", ["tfidf", "cointegrated/rubert-tiny2"],
                                   format_func=lambda e: "Быстрый (TF-IDF)" if e == "tfidf" else "Нейросеть rubert-tiny2")
    use_critic = st.toggle("Агент-критик проверяет рекомендации", value=True)
    if provider == "demo":
        st.info("Демо-режим: аспекты ищутся словарём, рекомендации шаблонные. Для настоящего анализа выберите модель и задайте ключ в переменных окружения.")

# --- Файлы ---------------------------------------------------------------
use_demo = st.checkbox("Взять демонстрационные данные", value=False)
inputs_spec: list[tuple[str, Path, bool]] = []
if use_demo:
    inputs_spec = DEMO
else:
    own_file = st.file_uploader("Отзывы своего товара (CSV или XLSX)", type=["csv", "xlsx"])
    own_name = st.text_input("Название своего товара", value="Свой товар")
    comp_files = st.file_uploader("Отзывы конкурентов (до трёх файлов)", type=["csv", "xlsx"], accept_multiple_files=True)
    tmp = Path(tempfile.gettempdir()) / "review_agent_uploads"
    tmp.mkdir(exist_ok=True)
    if own_file:
        p = tmp / f"own_{own_file.name}"
        p.write_bytes(own_file.getvalue())
        inputs_spec.append((own_name, p, True))
    for k, f in enumerate(comp_files[:3], start=1):
        p = tmp / f"comp{k}_{f.name}"
        p.write_bytes(f.getvalue())
        inputs_spec.append((f"Конкурент {k}: {Path(f.name).stem}", p, False))


def mapping_editor(name: str, path: Path, key: str) -> ColumnMapping:
    """Показывает первые строки файла и даёт поправить сопоставление колонок (FR-1)."""
    df = read_table(path)
    guessed = guess_mapping(list(df.columns))
    with st.expander(f"{name}: {len(df)} строк, колонки", expanded=False):
        st.dataframe(df.head(5), use_container_width=True)
        options = [NONE, *df.columns]
        cols = st.columns(4)
        values = {}
        for i, (fld, label) in enumerate(FIELDS.items()):
            current = getattr(guessed, fld)
            values[fld] = cols[i % 4].selectbox(label, options, index=options.index(current) if current in options else 0, key=f"{key}-{fld}")
    return ColumnMapping(**{k: (None if v == NONE else v) for k, v in values.items()})


mappings = [mapping_editor(name, path, f"m{k}") for k, (name, path, _) in enumerate(inputs_spec)]

if inputs_spec and not any(own for _, _, own in inputs_spec):
    st.warning("Загрузите отзывы своего товара.")

run = st.button("Запустить анализ", type="primary", disabled=not any(own for _, _, own in inputs_spec))

# --- Запуск --------------------------------------------------------------
if run:
    settings = Settings.from_env()
    try:
        client = demo_client() if provider == "demo" else make_client(Settings(llm_provider=provider, llm_model=model))
    except Exception as e:  # noqa: BLE001 — показываем пользователю понятную ошибку
        st.error(f"Не удалось подключить модель: {e}")
        st.stop()
    inputs = []
    comp_no = 0
    for (name, path, own), mapping in zip(inputs_spec, mappings):
        if own:
            pid = "own"
        else:
            comp_no += 1
            pid = f"comp{comp_no}"
        inputs.append(ProductInput(Product(id=pid, name=name, is_own=own), path=path, mapping=mapping))
    embedder = TfidfEmbedder() if embedder_choice == "tfidf" else SentenceTransformerEmbedder(embedder_choice)
    with st.status("Анализ отзывов…", expanded=True) as status:
        try:
            result = run_analysis(
                inputs, LLMGateway(client, temperature=settings.temperature), settings,
                taxonomy=None if taxonomy_choice == "auto" else load_taxonomy(taxonomy_choice),
                embedder=embedder, storage=Storage(ROOT / settings.db_path), use_critic=use_critic,
                progress=st.write,
            )
        except Exception as e:  # noqa: BLE001
            status.update(label="Ошибка", state="error")
            st.exception(e)
            st.stop()
        status.update(label="Готово", state="complete", expanded=False)
    st.session_state["result"] = result
    st.session_state["demo"] = provider == "demo"

# --- Отчёт ---------------------------------------------------------------
result: AnalysisResult | None = st.session_state.get("result")
if result:
    for w in result.warnings:
        st.warning(w)
    if st.session_state.get("demo"):
        st.info("Это демонстрационный запуск без языковой модели.")

    mention_by_id = {m.id: m for m in result.mentions}
    sentence_by_id = {s.id: s for s in result.sentences}
    review_by_id = {r.id: r for r in result.reviews}
    issue_by_id = {i.id: i for i in result.issues}
    n_own = sum(r.product_id in result.own_ids for r in result.reviews)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Отзывов своего товара", n_own)
    c2.metric("Проблем", len(result.issues))
    c3.metric("Рекомендаций", len(result.recommendations))
    cost = sum(v["cost"] for v in result.llm_summary.values())
    c4.metric("Стоимость", f"{cost:.2f} ₽")

    tabs = st.tabs(["Рекомендации", "Проблемы", "Атрибуты", "Сравнение", "Журнал"])

    with tabs[0]:
        if not result.recommendations:
            st.write("Рекомендаций, прошедших проверку, нет.")
        for k, rec in enumerate(result.recommendations, start=1):
            issue = issue_by_id.get(rec.issue_id or "")
            st.markdown(f"**{k}. {rec.action}**")
            st.caption(f"{rec.type.value} · сложность {rec.effort} из 3 · приоритет {rec.priority:.3f} · "
                       + (f"проблема «{issue.title}»" if issue else "сильная сторона"))
            st.write(rec.rationale)
            for q in rec.quotes:
                st.markdown(f"> «{q}»")

    with tabs[1]:
        for issue in sorted(result.issues, key=lambda i: -i.coverage):
            with st.expander(f"{issue.title} — {issue.coverage * 100:.0f} % отзывов, {len(issue.mention_ids)} упоминаний ({issue.attribute})"):
                # Переход к отзывам, на которых основан вывод
                for mid in issue.mention_ids:
                    m = mention_by_id[mid]
                    review = review_by_id[sentence_by_id[m.sentence_id].review_id]
                    stars = f"{review.rating}★ " if review.rating else ""
                    st.markdown(f"- {stars}**{m.quote}** — _{(review.text or review.cons)[:300]}_")

    with tabs[2]:
        rows = [{"Атрибут": s.attribute, "Охват": s.coverage, "Негатив": s.neg_share, "Позитив": s.pos_share,
                 "Средняя оценка": s.avg_rating} for s in result.stats if s.product_id in result.own_ids]
        st.dataframe(pd.DataFrame(rows).sort_values("Охват", ascending=False), hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.ProgressColumn(c, format="percent", min_value=0, max_value=1)
                                    for c in ("Охват", "Негатив", "Позитив")})

    with tabs[3]:
        if any(not p.is_own for p in result.products):
            st.caption("Доля отзывов с жалобами на атрибут")
            st.dataframe(pd.DataFrame([{"Атрибут": c.attribute, "Свой товар": c.own_rate, "Конкуренты": c.competitor_rate,
                                        "Итог": c.verdict} for c in result.comparison]), hide_index=True, use_container_width=True)
        else:
            st.write("Отзывы конкурентов не загружены.")

    with tabs[4]:
        st.dataframe(pd.DataFrame(result.llm_summary).T, use_container_width=True)
        st.write(f"Отклонено критиком: {len(result.rejected_by_critic)}, проверкой опоры на отзывы: {len(result.ungrounded)}")
        for rec, reasons in result.ungrounded:
            st.caption(f"«{rec.action}»: {'; '.join(reasons)}")
        if result.errors:
            st.error("\n".join(result.errors[:10]))

    st.download_button("Скачать отчёт (Markdown)", render_markdown(result), file_name="otchet.md", mime="text/markdown")
