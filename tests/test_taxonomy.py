"""Тесты справочника аспектов: модель подменена заглушкой."""
import json

from review_agent.ingest.loader import Отзыв
from review_agent.llm.base import FakeLLMClient
from review_agent.llm.gateway import LLMGateway
from review_agent.pipeline.taxonomy import Справочник, выбрать_выборку, построить_справочник
from review_agent.prompts import загрузить_промпт


def отзывы(n):
    return [Отзыв(id=str(i), product="Куртка", is_own=True, text=f"Отзыв номер {i}, куртка хорошая",
                  cons="маломерит" if i % 2 else "") for i in range(n)]


ОТВЕТ = json.dumps({"attributes": [
    {"name": "Размер и посадка", "features": ["маломерит", "по фигуре"]},
    {"name": "размер и посадка", "features": ["большемерит", "маломерит"]},   # повтор
    {"name": "Ткань", "features": ["мягкая"]},
    {"name": "Доставка", "features": ["быстро"]}]}, ensure_ascii=False)


def шлюз(*ответы):
    return LLMGateway(FakeLLMClient(list(ответы)), sleep=lambda _: None)


def test_выборка_воспроизводима_и_ограничена():
    о = отзывы(300)
    а, б = выбрать_выборку(о, 150, seed=1), выбрать_выборку(о, 150, seed=1)
    assert [x.id for x in а] == [x.id for x in б] and len(а) == 150
    assert len(выбрать_выборку(отзывы(10), 150)) == 10


def test_справочник_строится_и_повторы_объединяются():
    с = построить_справочник(шлюз(ОТВЕТ), отзывы(20), категория="одежда")
    имена = [а.name for а in с.attributes]
    assert имена == ["Размер и посадка", "Ткань", "Доставка"]
    assert с.attributes[0].features == ["маломерит", "по фигуре", "большемерит"]


def test_промпт_содержит_категорию_и_отзывы():
    клиент = FakeLLMClient([ОТВЕТ])
    построить_справочник(LLMGateway(клиент, sleep=lambda _: None), отзывы(5), категория="одежда")
    текст = клиент.calls[0][0].content
    assert "одежда" in текст and "Отзыв номер" in текст and "{" in текст  # JSON-пример на месте
    assert "{категория}" not in текст and "{отзывы}" not in текст


def test_сохранение_и_загрузка(tmp_path):
    с = построить_справочник(шлюз(ОТВЕТ), отзывы(20), категория="одежда")
    п = tmp_path / "taxonomy.json"
    с.сохранить(п)
    assert Справочник.загрузить(п) == с


def test_промпт_имеет_версию_в_имени():
    assert "Категория товара" in загрузить_промпт("taxonomy_v1")


def test_актуальный_промпт_v2_разделяет_упаковку_и_доставку():
    from review_agent.pipeline.taxonomy import ВЕРСИЯ_ПРОМПТА
    assert ВЕРСИЯ_ПРОМПТА == "taxonomy_v2"
    текст = загрузить_промпт(ВЕРСИЯ_ПРОМПТА)
    assert "РАЗНЫЕ атрибуты" in текст and "НЕ включай" in текст
    текст.format(категория="одежда", отзывы="1. тест")   # плейсхолдеры подставляются без ошибок
