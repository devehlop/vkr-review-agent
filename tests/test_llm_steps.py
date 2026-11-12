"""Тесты шагов с моделью и сквозной прогон конвейера на заглушке модели."""
import json

import pytest

from review_agent.llm import FakeClient, LLMGateway
from review_agent.models import Issue, Review, Sentence
from review_agent.pipeline.critic import review_draft
from review_agent.pipeline.issues import cluster_texts
from review_agent.pipeline.orchestrator import Settings, run_pipeline
from review_agent.pipeline.recommend import Draft, generate, issue_context
from review_agent.pipeline.taxonomy import build_taxonomy, review_text
from review_agent.report import build_report
from review_agent.storage import Storage
from tests.fake_model import ScriptedModel


def make_reviews() -> list[Review]:
    """35 отзывов на свой товар (6 жалоб на швы, 4 на размер) и 12 на товар конкурента."""
    own = ["Швы расходятся после стирки."] * 0
    own += [f"Швы расходятся после стирки номер {i}." for i in range(4)]
    own += ["Шов разошёлся на боку.", "Расходятся швы на рукаве, жаль."]
    own += [f"Сильно маломерит, берите больше, заказ {i}." for i in range(4)]
    own += [f"Ткань приятная, ношу с удовольствием, покупка {i}." for i in range(25)]
    other = [f"Швы ровные, всё аккуратно, отзыв {i}." for i in range(8)] + [f"Ткань мягкая, {i}." for i in range(4)]
    return ([Review(i + 1, "Платье", text=t, rating=2 if "асход" in t or "азош" in t else 5) for i, t in enumerate(own)]
            + [Review(i + 1, "Конкурент", text=t, rating=5, is_own=False) for i, t in enumerate(other)])


def gateway_for(model: ScriptedModel) -> LLMGateway:
    return LLMGateway(FakeClient(model), format_retries=1)


def test_review_text_joins_fields():
    assert review_text(Review(1, "p", text="Хорошо")) == "Хорошо"
    assert review_text(Review(1, "p", text="Ок", cons="Маломерит")) == "Недостатки: Маломерит Комментарий: Ок"


def test_build_taxonomy_and_validation():
    taxonomy = build_taxonomy(gateway_for(ScriptedModel()), make_reviews())
    hints = taxonomy.as_hints()
    assert list(hints)[:2] == ["Качество пошива", "Размер и посадка"]
    assert hints["Доставка"] == "пояснение (например: признак)"
    # слишком мало атрибутов: ответ отклоняется, после повтора принимается
    short = json.dumps({"attributes": [{"name": "Цена"}]}, ensure_ascii=False)
    good = json.dumps({"attributes": [{"name": f"Атрибут {i}"} for i in range(5)]}, ensure_ascii=False)
    client = FakeClient([short, good])
    assert len(build_taxonomy(LLMGateway(client), make_reviews()).attributes) == 5
    assert "от 5 до 12" in client.calls[1][-1]["content"]


def test_cluster_texts_groups_similar_complaints():
    texts = ["швы расходятся", "расходятся швы на боку", "швы расходятся после стирки",
             "сильно маломерит", "маломерит на размер", "очень маломерит", "ткань просвечивает"]
    groups = cluster_texts(texts)
    assert sorted(map(sorted, groups)) == [[0, 1, 2], [3, 4, 5], [6]]
    assert cluster_texts([]) == [] and cluster_texts(["один"]) == [[0]]


def test_tfidf_baseline_misses_other_word_forms_but_embedder_is_pluggable():
    """Известное ограничение базового варианта: другая форма слова («шов разошёлся») не попадает
    в группу «швы расходятся». Нейросетевые эмбеддинги подключаются параметром embed."""
    texts = ["швы расходятся", "расходятся швы на боку", "шов разошёлся"]
    assert sorted(map(sorted, cluster_texts(texts))) == [[0, 1], [2]]

    def by_meaning(items):  # условные «смысловые» векторы: все три фразы про швы
        import numpy as np
        return np.array([[1.0, 0.1 * i] for i, _ in enumerate(items)])

    assert cluster_texts(texts, embed=by_meaning) == [[0, 1, 2]]


def test_generate_rejects_invented_quote():
    sentences = [Sentence(1, 1, "Швы расходятся."), Sentence(2, 2, "Шов разошёлся."), Sentence(3, 3, "Расходятся швы.")]
    issue = Issue(1, "Платье", "Качество пошива", "Расходятся швы", [1, 2, 3], 0.2, 0.8)
    context = issue_context(issue, sentences, None, "одежда")
    assert "1. Швы расходятся." in context["sentences"] and context["coverage"] == "20 %"
    drafts = generate(gateway_for(ScriptedModel()), context)
    assert drafts[0].quotes[0].sentence_id == 1 and drafts[0].effort == 2
    from review_agent.llm import FormatError
    with pytest.raises(FormatError, match="нет в предложении 1"):
        generate(gateway_for(ScriptedModel(invent_quote=True)), context)


def _draft_and_context():
    sentences = [Sentence(i, i, "Швы расходятся.") for i in (1, 2, 3)]
    issue = Issue(1, "Платье", "Качество пошива", "Расходятся швы", [1, 2, 3], 0.2, 0.8)
    draft = Draft(action="Улучшить качество", type="товар", rationale="жалуются", effort=3,
                  quotes=[{"sentence_id": 1, "quote": "Швы расходятся"}])
    return draft, issue_context(issue, sentences, None, "одежда")


def test_critic_accepts_after_revision():
    draft, context = _draft_and_context()
    model = ScriptedModel(critic_scores=[2, 5])
    reviewed = review_draft(gateway_for(model), context, draft, [])
    assert reviewed.accepted and reviewed.draft.action == "Исправленное действие"
    assert [(v.iteration, v.passed) for v in reviewed.verdicts] == [(0, False), (1, True)]
    assert model.steps == ["critic", "recommend", "critic"]


def test_critic_rejects_after_two_revisions():
    draft, context = _draft_and_context()
    model = ScriptedModel(critic_scores=[2])
    reviewed = review_draft(gateway_for(model), context, draft, [])
    assert not reviewed.accepted and len(reviewed.verdicts) == 3
    assert model.steps.count("recommend") == 2  # не больше двух доработок


def test_pipeline_end_to_end(tmp_path):
    model = ScriptedModel()
    storage = Storage(tmp_path / "база.db")
    steps = []
    result = run_pipeline(gateway_for(model), make_reviews(), "Платье", storage=storage, on_step=steps.append)
    assert steps[0] == "Предобработка отзывов" and "Рекомендации: проблема 2 из 2" in steps

    assert result.warnings == [] and result.clean.kept == 47
    # жалоб на швы шесть, но «Шов разошёлся на боку» базовая кластеризация в группу не относит
    assert [(i.attribute, len(i.sentence_ids)) for i in result.issues] == [("Качество пошива", 5), ("Размер и посадка", 4)]
    sewing = result.issues[0]
    assert sewing.coverage == pytest.approx(5 / 35, abs=1e-4) and sewing.rarer_at_competitors
    assert [r.issue_id for r in result.recommendations] == [1, 2]
    top = result.recommendations[0]
    assert top.grounded and top.priority == pytest.approx(5 / 35 * 1.0 * 1.5 / 2, abs=1e-3)
    assert result.rejected == [] and len(result.critiques) == 2
    assert model.steps[0] == "taxonomy" and model.steps.count("extract") == 3  # 47 предложений пакетами по 20

    # результат сохранён в базе
    assert len(storage.load_recommendations(result.run_id)) == 2
    assert len(storage.load_critiques(result.run_id)) == 2
    assert storage.run_status(result.run_id) == "готово"

    report = build_report(result)
    assert "# Отчёт по отзывам: Платье" in report
    assert "**Ткань и материал**" in report  # сильная сторона
    assert "| Качество пошива | 100 % | 0 % | хуже |" in report
    assert "### 1. Усилить швы двойной строчкой" in report and "«Швы расходятся после стирки номер 0.»" in report


def test_pipeline_without_critic_and_with_ready_taxonomy():
    model = ScriptedModel()
    taxonomy = {"Качество пошива": "", "Размер и посадка": "", "Ткань и материал": ""}
    result = run_pipeline(gateway_for(model), make_reviews(), "Платье", taxonomy, Settings(use_critic=False))
    assert "taxonomy" not in model.steps and "critic" not in model.steps
    assert len(result.recommendations) == 2 and result.critiques == []


def test_pipeline_critic_rejection_goes_to_journal_only():
    result = run_pipeline(gateway_for(ScriptedModel(critic_scores=[1])), make_reviews(), "Платье")
    assert result.recommendations == []
    assert len(result.rejected) == 2 and "не прошла проверку критика" in result.rejected[0][1]
    assert "Рекомендаций, прошедших проверку, нет." in build_report(result)


def test_pipeline_warns_on_few_reviews_and_survives_bad_generation():
    reviews = [r for r in make_reviews() if r.product == "Платье"][:12]
    result = run_pipeline(gateway_for(ScriptedModel(invent_quote=True)), reviews, "Платье")
    assert any("Отзывов на товар всего 12" in w for w in result.warnings)
    assert any("рекомендации не получены" in w for w in result.warnings)
    assert result.recommendations == [] and len(result.issues) == 2
    with pytest.raises(ValueError, match="Нет отзывов"):
        run_pipeline(gateway_for(ScriptedModel()), reviews, "Другой товар")
