"""Тесты подозрительных отзывов, советов по сильным сторонам, резюме сравнения, эмбеддингов и агента вопросов."""
import json

import httpx
import pytest

from review_agent.llm import FakeClient, LLMError, LLMGateway
from review_agent.llm.openai_compat import OpenAICompatClient
from review_agent.models import Review
from review_agent.pipeline.orchestrator import Settings, run_pipeline
from review_agent.preprocess import mark_suspicious
from review_agent.qa import answer_question
from review_agent.report import build_report
from review_agent.storage import Storage
from tests.fake_model import ScriptedModel
from tests.test_llm_steps import gateway_for, make_reviews


def test_mark_suspicious_near_duplicates_only():
    reviews = [
        Review(1, "p", "Отличное платье, ткань приятная, размер подошёл, рекомендую всем!"),
        Review(2, "p", "Отличное платье, ткань приятная, размер подошёл, рекомендую всем!!!"),
        Review(3, "p", "Швы расходятся после первой стирки, очень разочарована покупкой."),
        Review(4, "p", "Всё отлично"),
        Review(5, "p", "Всё отлично!"),  # короткие отзывы не проверяются
        Review(6, "другой", "Отличное платье, ткань приятная, размер подошёл, рекомендую всем!"),  # другой товар
    ]
    assert mark_suspicious(reviews) == 2
    assert [r.suspicious for r in reviews] == [True, True, False, False, False, False]


def test_exclude_suspicious_setting():
    kept = run_pipeline(gateway_for(ScriptedModel()), make_reviews(), "Платье")
    dropped = run_pipeline(gateway_for(ScriptedModel()), make_reviews(), "Платье", settings=Settings(exclude_suspicious=True))
    assert len(kept.reviews) == 47 and "учтены в анализе" in kept.warnings[0]
    assert len(dropped.reviews) < 47 and "исключены из анализа" in dropped.warnings[0]
    assert all(not r.suspicious for r in dropped.reviews)


def test_card_tips_and_comparison_summary_in_result_and_report(tmp_path):
    storage = Storage(tmp_path / "база.db")
    result = run_pipeline(gateway_for(ScriptedModel()), make_reviews(), "Платье", storage=storage)
    tip = result.card_tips[0]
    assert tip.attribute == "Ткань и материал" and tip.quotes[0][1].startswith("Ткань приятная")
    assert result.comparison_summary == "По характеристике «Качество пошива» товар уступает: 100 % негатива против 0 % у конкурента."
    report = build_report(result)
    assert "  - Для карточки товара: Добавить в описание слова покупателей о ткани" in report
    assert result.comparison_summary in report
    assert any(r.suspicious for r in storage.load_reviews(result.run_id))


def test_summary_with_invented_number_is_dropped():
    result = run_pipeline(gateway_for(ScriptedModel(invent_number=True)), make_reviews(), "Платье")
    assert result.comparison_summary is None  # число 55 % отсутствует в таблице: резюме в отчёт не попадает
    assert "## Сравнение с конкурентами" in build_report(result)


def test_extras_can_be_switched_off():
    model = ScriptedModel()
    result = run_pipeline(gateway_for(model), make_reviews(), "Платье",
                          settings=Settings(card_tips=False, compare_summary=False))
    assert result.card_tips == [] and result.comparison_summary is None
    assert "strength" not in model.steps and "compare_summary" not in model.steps


# --- эмбеддинги от модели ---
def embedding_client(handler) -> OpenAICompatClient:
    client = OpenAICompatClient("qwen", "http://localhost:1/v1", "m")
    client.embed_model = "emb"
    client._http = httpx.Client(transport=httpx.MockTransport(handler))
    return client


def test_client_embed_batches_and_orders():
    sizes = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/v1/embeddings" and body["model"] == "emb"
        sizes.append(len(body["input"]))
        data = [{"index": i, "embedding": [float(len(t)), 1.0]} for i, t in enumerate(body["input"])]
        return httpx.Response(200, json={"data": list(reversed(data))})  # сервер может вернуть в другом порядке

    vectors = embedding_client(handler).embed(["а", "бб", "ввв"], batch_size=2)
    assert sizes == [2, 1] and vectors == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]


def test_embed_without_model_fails():
    client = OpenAICompatClient("qwen", "http://localhost:1/v1", "m")
    with pytest.raises(LLMError, match="не задана"):
        client.embed(["текст"])


class EmbeddingFake(FakeClient):
    """Заглушка модели с эмбеддингами: все жалобы на швы получают близкие векторы."""

    embed_model = "emb"
    fail = False

    def embed(self, texts):
        if self.fail:
            raise LLMError("сервер эмбеддингов недоступен")
        return [[1.0, 0.0] if ("шв" in t.lower() or "шов" in t.lower()) else [0.0, 1.0] for t in texts]


def test_model_embeddings_merge_word_forms_and_fall_back():
    settings = Settings(embeddings="model")
    merged = run_pipeline(LLMGateway(EmbeddingFake(ScriptedModel()), format_retries=1), make_reviews(), "Платье", settings=settings)
    # с эмбеддингами «Шов разошёлся на боку» попадает в общую группу: шесть жалоб, а не пять
    assert len(merged.issues[0].sentence_ids) == 6

    broken = EmbeddingFake(ScriptedModel())
    broken.fail = True
    fallback = run_pipeline(LLMGateway(broken, format_retries=1), make_reviews(), "Платье", settings=settings)
    assert len(fallback.issues[0].sentence_ids) == 5
    assert any("Эмбеддинги модели недоступны" in w for w in fallback.warnings)

    plain = run_pipeline(gateway_for(ScriptedModel()), make_reviews(), "Платье", settings=settings)
    assert any("Модель эмбеддингов не настроена" in w for w in plain.warnings)


# --- агент вопросов ---
@pytest.fixture(scope="module")
def analysis():
    return run_pipeline(gateway_for(ScriptedModel()), make_reviews(), "Платье")


def test_agent_searches_then_answers_with_citations(analysis):
    model = ScriptedModel()
    answer = answer_question(gateway_for(model), analysis, "Что пишут про швы?")
    assert answer.ok and answer.text == "Покупатели пишут об этом в отзывах."
    assert answer.tool_calls == [("search_reviews", {"query": "Что пишут про швы?"})]
    texts = {s.id: s.text for s in analysis.sentences}
    assert len(answer.sentence_ids) == 2 and all("шв" in texts[i].lower() for i in answer.sentence_ids)


def test_agent_can_chain_tools(analysis):
    answer = answer_question(gateway_for(ScriptedModel()), analysis, "Покажи статистику по швам")
    assert [name for name, _ in answer.tool_calls] == ["search_reviews", "aspect_stats"] and answer.ok


def test_agent_citation_of_unseen_sentence_is_rejected(analysis):
    client = FakeClient(ScriptedModel(invent_citation=True))
    answer = answer_question(LLMGateway(client, format_retries=1), analysis, "Что пишут про швы?")
    assert answer.ok and 9999 not in answer.sentence_ids  # после замечания модель исправила ссылки
    assert any("инструменты не возвращали" in m["content"] for call in client.calls for m in call)


def test_agent_tools_directly(analysis):
    from review_agent.qa.agent import Toolbox

    tools = Toolbox(analysis)
    assert "Найдено предложений" in tools.search_reviews("маломерит", attribute="Размер и посадка", sentiment=-1)
    assert tools.search_reviews("абракадабра ъъъ").startswith("По словам запроса ничего не найдено")
    # поиск по словам не понимает смысл: «размер» не находит «маломерит», пока не указана характеристика
    assert tools.search_reviews("размер").startswith("По словам запроса ничего не найдено")
    assert "маломерит" in tools.search_reviews("размер", attribute="Размер и посадка")
    assert tools.search_reviews("швы", product="Конкурент").count("Швы ровные") == 8
    assert "доля негатива 100 %" in tools.aspect_stats("Качество пошива")
    assert "упоминаний нет" in tools.aspect_stats("Цена")
    assert "наш товар хуже" in tools.compare("Качество пошива")
    assert tools.list_issues().startswith("1. Жалобы: качество пошива")
    assert tools.call("compare", {"лишний": 1}).startswith("Ошибка вызова compare")


def test_agent_search_uses_model_embeddings_when_available(analysis):
    from review_agent.qa.agent import Toolbox

    def embed(texts):  # условные «смысловые» векторы: запрос про размер близок к «маломерит»
        return [[1.0, 0.0] if ("размер" in t.lower() or "маломерит" in t.lower()) else [0.0, 1.0] for t in texts]

    found = Toolbox(analysis, embed).search_reviews("размер", limit=4)
    assert found.count("маломерит") == 4


def test_agent_reports_failure():
    result = run_pipeline(gateway_for(ScriptedModel()), make_reviews(), "Платье")
    answer = answer_question(LLMGateway(FakeClient(["не JSON"]), format_retries=1), result, "Вопрос")
    assert not answer.ok and "Не удалось получить ответ" in answer.text
