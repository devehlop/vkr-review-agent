"""Веб-интерфейс для продавца (FR-19).

Запуск:
    streamlit run app/streamlit_app.py
"""
import json

import pandas as pd
import streamlit as st

from review_agent import config
from review_agent.ingest import guess_mapping, read_table, reviews_from_table
from review_agent.llm.base import LLMError
from review_agent.llm.gateway import LLMGateway
from review_agent.pipeline.compare import NO_DATA
from review_agent.pipeline.orchestrator import PipelineResult, Settings, run_pipeline
from review_agent.report import build_report
from review_agent.storage import Storage

FIELDS = {"text": "Текст отзыва", "rating": "Оценка", "created": "Дата",
          "pros": "Достоинства", "cons": "Недостатки", "variant": "Вариант товара"}
MODELS = {"gigachat": "GigaChat", "yandex": "YandexGPT", "qwen": "Qwen (открытая модель)"}
EFFORT = {1: "низкая", 2: "средняя", 3: "высокая"}
SKIP = "нет"
MAX_COMPETITORS = 3

st.set_page_config(page_title="Анализ отзывов", page_icon="🧵", layout="wide")


def pct(share: float) -> str:
    return f"{share * 100:.0f} %"


def upload_block(key: str, title: str, default_name: str) -> tuple[str, pd.DataFrame, dict] | None:
    """Загрузка одного файла с отзывами и сопоставление колонок. Возвращает (название, таблица, соответствие)."""
    name = st.text_input("Название товара", value=default_name, key=f"{key}_name")
    file = st.file_uploader(title, type=["csv", "xlsx"], key=f"{key}_file")
    if file is None:
        return None
    try:
        table = read_table(file, file.name)
    except Exception as error:  # файл повреждён или в неожиданном формате
        st.error(f"Не удалось прочитать файл: {error}")
        return None
    guessed = guess_mapping(list(table.columns))
    options = [SKIP] + [str(c) for c in table.columns]
    mapping = {}
    with st.expander(f"Колонки файла: строк {len(table)}", expanded="text" not in guessed and "cons" not in guessed):
        st.caption("Проверьте, какие колонки использовать. Остальные колонки (имя покупателя, фото) не читаются.")
        columns = st.columns(3)
        for index, (field, label) in enumerate(FIELDS.items()):
            default = options.index(str(guessed[field])) if field in guessed else 0
            choice = columns[index % 3].selectbox(label, options, index=default, key=f"{key}_{field}")
            if choice != SKIP:
                mapping[field] = choice
        if mapping:  # показываем только выбранные колонки
            st.dataframe(table[list(dict.fromkeys(mapping.values()))].head(5), width="stretch", hide_index=True)
    if not any(f in mapping for f in ("text", "pros", "cons")):
        st.warning("Укажите колонку с текстом отзыва, достоинствами или недостатками.")
        return None
    return name.strip() or default_name, table, mapping


def run_analysis(own, competitors, model_name: str, settings: Settings, taxonomy: dict | None) -> PipelineResult:
    reviews = reviews_from_table(own[1], own[0], own[2])
    for name, table, mapping in competitors:
        reviews += reviews_from_table(table, name, mapping, is_own=False)
    client = config.make_client(model_name)
    gateway = LLMGateway(client, journal_path=config.ROOT / "logs" / f"{client.name}.jsonl",
                         cache_dir=config.ROOT / ".cache" / client.name)
    with st.status("Идёт анализ…", expanded=True) as status:
        result = run_pipeline(gateway, reviews, own[0], taxonomy, settings,
                              Storage(config.ROOT / "data" / "review_agent.db"), on_step=st.write)
        status.update(label="Анализ завершён", state="complete", expanded=False)
    return result


def show_reviews(result: PipelineResult, sentence_ids: list[int], highlight: dict[int, str] | None = None) -> None:
    """Показывает отзывы целиком для выбранных предложений: переход от вывода к примерам."""
    sentence_by_id = {s.id: s for s in result.sentences}
    review_by_id = {r.id: r for r in result.reviews}
    for sentence_id in sentence_ids:
        sentence = sentence_by_id.get(sentence_id)
        if sentence is None:
            continue
        review = review_by_id[sentence.review_id]
        stars = f"{'★' * review.rating}{'☆' * (5 - review.rating)} " if review.rating else ""
        when = f"{review.created:%d.%m.%Y} " if review.created else ""
        full = " ".join(t for t in (review.pros, review.cons, review.text) if t)
        with st.container(border=True):
            st.markdown(f"**«{sentence.text}»**")
            if full != sentence.text:
                st.caption(f"{stars}{when}Отзыв целиком: {full}")
            elif stars or when:
                st.caption(f"{stars}{when}")


def tab_report(result: PipelineResult) -> None:
    own = result.own_product
    issue_by_id = {i.id: i for i in result.issues}
    numbers = st.columns(4)
    numbers[0].metric("Отзывов на товар", sum(r.product == own for r in result.reviews))
    numbers[1].metric("Проблем", len(result.issues))
    numbers[2].metric("Рекомендаций", len(result.recommendations))
    numbers[3].metric("Конкурентов", len({r.product for r in result.reviews if r.product != own}))

    st.subheader("Рекомендации по приоритету")
    if not result.recommendations:
        st.info("Рекомендаций, прошедших проверку, нет.")
    for number, rec in enumerate(result.recommendations, start=1):
        issue = issue_by_id[rec.issue_id]
        with st.expander(f"{number}. {rec.action}", expanded=number == 1):
            st.markdown(f"**Проблема:** {issue.title} ({pct(issue.coverage)} отзывов)"
                        + (" · у конкурентов встречается заметно реже" if issue.rarer_at_competitors else ""))
            st.markdown(f"**Обоснование:** {rec.rationale}")
            if rec.expected_effect:
                st.markdown(f"**Ожидаемый эффект:** {rec.expected_effect}")
            st.caption(f"Тип: {rec.type} · Сложность: {EFFORT[rec.effort]} · Приоритет: {rec.priority}")
            st.markdown("**Отзывы, на которых основана рекомендация:**")
            show_reviews(result, [sid for sid, _ in rec.quotes])

    st.subheader("Сильные стороны")
    strong = [s for s in result.stats if s.product == own and not s.feature and s.mentions >= 5
              and s.positive / s.mentions >= 0.7]
    if strong:
        for item in sorted(strong, key=lambda s: -s.coverage):
            st.markdown(f"- **{item.attribute}**: упоминается в {pct(item.coverage)} отзывов, "
                        f"{pct(item.positive / item.mentions)} упоминаний положительные")
    else:
        st.write("Не выявлены.")
    st.download_button("Скачать отчёт (Markdown)", build_report(result), file_name="report.md", mime="text/markdown")


def tab_issues(result: PipelineResult) -> None:
    if not result.issues:
        st.info("Повторяющихся проблем не найдено.")
    for issue in result.issues:
        st.markdown(f"### {issue.id}. {issue.title}")
        st.caption(f"Характеристика: {issue.attribute} · встречается в {pct(issue.coverage)} отзывов · "
                   f"упоминаний: {len(issue.sentence_ids)}"
                   + (" · у конкурентов встречается заметно реже" if issue.rarer_at_competitors else ""))
        with st.expander(f"Показать отзывы ({len(issue.sentence_ids)})"):
            show_reviews(result, issue.sentence_ids)


def tab_aspects(result: PipelineResult) -> None:
    own = result.own_product
    rows = [s for s in result.stats if s.product == own and not s.feature]
    if not rows:
        st.info("Упоминаний характеристик не найдено.")
        return
    table = pd.DataFrame({
        "Характеристика": [s.attribute for s in rows],
        "Упоминаний": [s.mentions for s in rows],
        "Охват, %": [round(s.coverage * 100) for s in rows],
        "Негатив": [s.negative for s in rows],
        "Нейтрально": [s.neutral for s in rows],
        "Позитив": [s.positive for s in rows],
        "Доля негатива, %": [round(s.neg_share * 100) for s in rows],
        "Средняя оценка": [s.avg_rating for s in rows],
    })
    st.caption("Охват: доля отзывов, где характеристика упомянута. Доля негатива: среди упоминаний характеристики.")
    st.dataframe(table, width="stretch", hide_index=True)
    st.bar_chart(table.set_index("Характеристика")[["Негатив", "Нейтрально", "Позитив"]],
                 color=["#c0392b", "#95a5a6", "#27ae60"], horizontal=True)

    attribute = st.selectbox("Подробнее о характеристике", [s.attribute for s in rows])
    features = [s for s in result.stats if s.product == own and s.attribute == attribute and s.feature]
    if features:
        st.dataframe(pd.DataFrame({
            "Признак": [s.feature for s in features], "Упоминаний": [s.mentions for s in features],
            "Доля негатива, %": [round(s.neg_share * 100) for s in features]}),
            width="stretch", hide_index=True)
    months = next(s.by_month for s in rows if s.attribute == attribute)
    if len(months) > 1:
        st.caption("Доля негатива по месяцам, %")
        st.line_chart(pd.Series({m: round(neg / total * 100) for m, (neg, total) in months.items()}))
    review_product = {r.id: r.product for r in result.reviews}
    ids = [s.id for s in result.sentences if review_product.get(s.review_id) == own
           and any(m.attribute == attribute for m in result.mentions.get(s.id, []))]
    with st.expander(f"Показать отзывы ({len(ids)})"):
        show_reviews(result, ids[:50])
        if len(ids) > 50:
            st.caption(f"Показаны первые 50 из {len(ids)}.")


def tab_compare(result: PipelineResult) -> None:
    competitors = sorted({r.product for r in result.reviews if r.product != result.own_product})
    if not competitors:
        st.info("Отзывы конкурентов не загружены.")
        return
    rows = []
    for item in result.comparisons:
        row = {"Характеристика": item.attribute, "Наш товар, доля негатива, %": round(item.own_neg_share * 100)}
        for name in competitors:
            row[f"{name}, %"] = round(item.competitors[name][0] * 100) if name in item.competitors else None
        row["Наш товар"] = "мало данных для сравнения" if item.verdict == NO_DATA else item.verdict
        rows.append(row)
    st.caption("Сравнивается доля негативных упоминаний характеристики. Вывод делается, только если упоминаний достаточно.")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def tab_journal(result: PipelineResult) -> None:
    usage = result.usage
    numbers = st.columns(4)
    numbers[0].metric("Запросов к модели", usage.calls)
    numbers[1].metric("Токенов (ввод / вывод)", f"{usage.tokens_in} / {usage.tokens_out}")
    numbers[2].metric("Стоимость", f"{usage.cost:.2f}")
    numbers[3].metric("Время модели, с", f"{usage.seconds:.0f}")
    if usage.cached:
        st.caption(f"Ответов взято из кэша без обращения к модели: {usage.cached}. "
                   "Чтобы запросить модель заново, удалите папку .cache.")
    clean = result.clean
    st.write(f"Загружено отзывов: {clean.total}. Пустых: {clean.empty}, дубликатов: {clean.duplicates}, "
             f"в анализе: {clean.kept}. Предложений: {len(result.sentences)}.")
    st.write(f"Ответов модели с ошибкой формата: {usage.format_errors}. Запросов без ответа: {usage.failed}.")
    st.subheader("Отклонённые рекомендации")
    if result.rejected:
        st.dataframe(pd.DataFrame(result.rejected, columns=["Действие", "Причина"]), width="stretch", hide_index=True)
    else:
        st.write("Нет.")
    st.subheader("Справочник характеристик")
    st.dataframe(pd.DataFrame(result.taxonomy.items(), columns=["Характеристика", "Пояснение"]),
                 width="stretch", hide_index=True)


# ---------- страница ----------
st.title("Анализ отзывов покупателей")
st.caption("Загрузите отзывы на свой товар и, если есть, на товары конкурентов. "
           "Агент найдёт проблемы и предложит, что изменить в первую очередь.")

with st.sidebar:
    st.header("Настройки")
    model_name = st.selectbox("Модель", list(MODELS), format_func=MODELS.get)
    category = st.text_input("Категория товаров", value="одежда")
    ready_taxonomy = st.checkbox("Готовый справочник для одежды", value=False,
                                 help="Иначе справочник характеристик составит модель по вашим отзывам.")
    use_critic = st.checkbox("Проверять рекомендации критиком", value=True,
                             help="Отдельный запрос к модели оценивает каждую рекомендацию и возвращает слабые на доработку.")
    st.caption("Ключи моделей задаются в файле .env, см. .env.example.")

left, right = st.columns(2)
with left:
    st.subheader("Свой товар")
    own = upload_block("own", "Файл с отзывами (CSV или XLSX)", "Мой товар")
with right:
    st.subheader("Конкуренты")
    count = st.number_input("Сколько товаров конкурентов", 0, MAX_COMPETITORS, 0)
    competitors = [upload_block(f"comp{i}", f"Отзывы на товар конкурента {i + 1}", f"Конкурент {i + 1}")
                   for i in range(int(count))]

ready = own is not None and all(c is not None for c in competitors)
if st.button("Запустить анализ", type="primary", disabled=not ready):
    names = [own[0]] + [c[0] for c in competitors]
    if len(set(names)) != len(names):
        st.error("Названия товаров должны различаться.")
    else:
        taxonomy = None
        if ready_taxonomy:
            taxonomy = json.loads((config.ROOT / "eval" / "taxonomy_clothing.json").read_text(encoding="utf-8"))
        try:
            st.session_state["result"] = run_analysis(
                own, competitors, model_name, Settings(category=category, use_critic=use_critic), taxonomy)
        except (RuntimeError, ValueError, LLMError) as error:
            # нет ключа модели, в файле нет отзывов или модель недоступна
            st.session_state.pop("result", None)
            st.error(f"Анализ не выполнен: {error}")

result: PipelineResult | None = st.session_state.get("result")
if result:
    st.divider()
    st.header(f"Результат: {result.own_product}")
    for warning in result.warnings:
        st.warning(warning)
    tabs = st.tabs(["Отчёт", "Проблемы", "Характеристики", "Сравнение с конкурентами", "Журнал"])
    for tab, render in zip(tabs, (tab_report, tab_issues, tab_aspects, tab_compare, tab_journal)):
        with tab:
            render(result)
