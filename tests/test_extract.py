"""Тесты извлечения аспектов: модель подменена заглушкой."""
import json

from review_agent.llm.base import FakeLLMClient
from review_agent.llm.gateway import LLMGateway
from review_agent.pipeline.extract import извлечь_аспекты
from review_agent.pipeline.taxonomy import Атрибут, Справочник
from review_agent.preprocess.cleaner import Предложение

СПРАВОЧНИК = Справочник("одежда", [Атрибут("Размер и посадка", ["маломерит"]),
                                   Атрибут("Ткань", ["мягкая"]), Атрибут("Доставка", ["быстро"])])


def предл(i, text, source="text", review="r1"):
    return Предложение(review, i, text, source, "Куртка", True)


def шлюз(*ответы, повторов=1):
    return LLMGateway(FakeLLMClient([json.dumps(о, ensure_ascii=False) if isinstance(о, dict) else о
                                     for о in ответы]),
                      max_retries=повторов, sleep=lambda _: None)


def test_упоминания_привязываются_к_предложениям():
    п = [предл(0, "Маломерит.", "cons"), предл(1, "Ткань мягкая, доставили быстро.", "pros")]
    ответ = {"items": [
        {"i": 1, "mentions": [{"attribute": "Размер и посадка", "sentiment": "negative"}]},
        {"i": 2, "mentions": [{"attribute": "Ткань", "sentiment": "positive"},
                              {"attribute": "Доставка", "sentiment": "positive"}]}]}
    р = извлечь_аспекты(шлюз(ответ), п, СПРАВОЧНИК)
    assert [(м.sentence_idx, м.attribute, м.sentiment, м.source) for м in р.mentions] == [
        (0, "Размер и посадка", "negative", "cons"), (1, "Ткань", "positive", "pros"),
        (1, "Доставка", "positive", "pros")]
    assert р.mentions[0].text == "Маломерит."


def test_пакеты_делятся_по_размеру_и_номера_локальные():
    п = [предл(i, f"Предложение {i}") for i in range(3)]
    ответ = {"items": [{"i": 1, "mentions": [{"attribute": "Ткань", "sentiment": "neutral"}]}]}
    р = извлечь_аспекты(шлюз(ответ, ответ), п, СПРАВОЧНИК, размер_пакета=2)
    assert р.stats["batches"] == 2
    assert [м.sentence_idx for м in р.mentions] == [0, 2]   # i=1 в каждом пакете


def test_атрибут_не_из_справочника_отбрасывается_регистр_не_важен():
    ответ = {"items": [{"i": 1, "mentions": [{"attribute": "ткань", "sentiment": "positive"},
                                             {"attribute": "Цвет", "sentiment": "positive"}]}]}
    р = извлечь_аспекты(шлюз(ответ), [предл(0, "Ткань хорошая, цвет красивый")], СПРАВОЧНИК)
    assert [м.attribute for м in р.mentions] == ["Ткань"] and р.stats["unknown_attribute"] == 1


def test_неверный_номер_предложения_и_повторы_упоминаний():
    ответ = {"items": [{"i": 9, "mentions": [{"attribute": "Ткань", "sentiment": "positive"}]},
                       {"i": 1, "mentions": [{"attribute": "Ткань", "sentiment": "positive"},
                                             {"attribute": "Ткань", "sentiment": "positive"}]}]}
    р = извлечь_аспекты(шлюз(ответ), [предл(0, "Ткань.")], СПРАВОЧНИК)
    assert len(р.mentions) == 1 and р.stats["bad_index"] == 1


def test_сбойный_пакет_делится_пополам_и_обрабатывается():
    п = [предл(0, "А."), предл(1, "Б.")]
    ответ1 = {"items": [{"i": 1, "mentions": [{"attribute": "Ткань", "sentiment": "positive"}]}]}
    # целый пакет: невалидный JSON (1 попытка); затем половины проходят по одному
    р = извлечь_аспекты(шлюз("обрезанный {", ответ1, ответ1), п, СПРАВОЧНИК, размер_пакета=2)
    assert р.stats["batches_split"] == 1 and р.stats["batches_failed"] == 0
    assert [м.sentence_idx for м in р.mentions] == [0, 1]


def test_предложение_не_прошедшее_даже_поодиночке_пропускается():
    п = [предл(0, "А."), предл(1, "Б.")]
    хороший = {"items": [{"i": 1, "mentions": [{"attribute": "Ткань", "sentiment": "positive"}]}]}
    # целый пакет и первое предложение сбоят, второе проходит
    р = извлечь_аспекты(шлюз("плохо", "плохо", хороший), п, СПРАВОЧНИК, размер_пакета=2)
    assert р.stats["batches_failed"] == 1 and р.stats["sentences_skipped"] == 1
    assert [м.sentence_idx for м in р.mentions] == [1]


def test_неизвестная_тональность_ломает_схему():
    плохой = {"items": [{"i": 1, "mentions": [{"attribute": "Ткань", "sentiment": "хорошо"}]}]}
    р = извлечь_аспекты(шлюз(плохой), [предл(0, "Ткань.")], СПРАВОЧНИК)
    assert р.stats["batches_failed"] == 1 and р.mentions == []


def test_промпт_содержит_справочник_и_метку_источника():
    клиент = FakeLLMClient([json.dumps({"items": []})])
    извлечь_аспекты(LLMGateway(клиент, sleep=lambda _: None),
                    [предл(0, "Маломерит.", "cons")], СПРАВОЧНИК)
    текст = клиент.calls[0][0].content
    assert "Размер и посадка: маломерит" in текст and "1. [минусы] Маломерит." in текст
