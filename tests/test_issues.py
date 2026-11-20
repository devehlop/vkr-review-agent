"""Тесты шага 6 на заглушках эмбеддингов и модели."""
import json

from review_agent.llm.base import FakeLLMClient
from review_agent.llm.embeddings import FakeEmbedder
from review_agent.llm.gateway import LLMGateway
from review_agent.pipeline.extract import Упоминание
from review_agent.pipeline.issues import выделить_проблемы


def м(review, attr, sent, text, is_own=True):
    return Упоминание(review, 0, attr, sent, "cons", text, "Куртка", is_own)


УПОМИНАНИЯ = [
    м("1", "Фурнитура", "negative", "Молния заедает при застёгивании"),
    м("2", "Фурнитура", "negative", "Молния заедает после стирки"),
    м("3", "Фурнитура", "negative", "Молния заедает сильно"),
    м("4", "Цвет", "negative", "Цвет темнее чем на фото"),          # кластер из 1: мал
    м("5", "Фурнитура", "positive", "Молния крепкая"),              # не негатив
    м("6", "Фурнитура", "negative", "Молния заедает всегда", is_own=False),   # конкурент
]


def шлюз(*ответы, повторов=1):
    return LLMGateway(FakeLLMClient(list(ответы)), max_retries=повторов, sleep=lambda _: None)


def запустить(g, **kw):
    return выделить_проблемы(g, FakeEmbedder(), УПОМИНАНИЯ, категория="одежда",
                             всего_отзывов=10, порог=0.6, **kw)


def test_проблема_из_негативных_упоминаний_своего_товара():
    р = запустить(шлюз('{"title": "Молния заедает"}'))
    assert len(р.problems) == 1
    п = р.problems[0]
    assert (п.attribute, п.title, п.mentions, п.reviews) == ("Фурнитура", "Молния заедает", 3, 3)
    assert п.coverage == 0.3 and п.review_ids == ["1", "2", "3"] and п.title_ok
    assert all("заедает" in ц for ц in п.quotes)


def test_малые_кластеры_не_становятся_проблемами():
    р = запустить(шлюз('{"title": "Молния заедает"}'))
    assert р.stats["small_clusters"] == 1 and р.stats["mentions_unassigned"] == 1
    assert р.stats["negative_mentions"] == 4        # без позитивного и без конкурента


def test_мин_размер_можно_снизить():
    р = запустить(шлюз('{"title": "Молния заедает"}', '{"title": "Цвет не как на фото"}'), мин_размер=1)
    assert {п.attribute for п in р.problems} == {"Фурнитура", "Цвет"}


def test_сбой_названия_не_роняет_шаг():
    р = запустить(шлюз("не json"))
    п = р.problems[0]
    assert not п.title_ok and "Фурнитура" in п.title and р.stats["titles_failed"] == 1


def test_без_негатива_модель_не_вызывается():
    клиент = FakeLLMClient([])
    эмб = FakeEmbedder()
    р = выделить_проблемы(LLMGateway(клиент, sleep=lambda _: None), эмб,
                          [м("1", "Цвет", "positive", "Красивый")], категория="одежда", всего_отзывов=5)
    assert р.problems == [] and клиент.calls == [] and эмб.calls == 0


def test_промпт_содержит_цитаты_и_атрибут():
    клиент = FakeLLMClient([json.dumps({"title": "Молния заедает"})])
    выделить_проблемы(LLMGateway(клиент, sleep=lambda _: None), FakeEmbedder(), УПОМИНАНИЯ,
                      категория="одежда", всего_отзывов=10, порог=0.6)
    текст = клиент.calls[0][0].content
    assert "Фурнитура" in текст and "- Молния заедает" in текст and "{" in текст


def test_цитаты_без_повторов_а_счёт_по_всем_упоминаниям():
    повторы = [м("1", "Доставка", "negative", "Доставка шла две недели"),
               м("2", "Доставка", "negative", "доставка шла две недели."),
               м("3", "Доставка", "negative", "Доставка шла две недели"),
               м("4", "Доставка", "negative", "Курьер опоздал на неделю")]
    клиент = FakeLLMClient(['{"title": "Задержка доставки"}'])
    р = выделить_проблемы(LLMGateway(клиент, sleep=lambda _: None), FakeEmbedder(), повторы,
                          категория="одежда", всего_отзывов=10, порог=1.5)
    п = р.problems[0]
    assert (п.mentions, п.reviews) == (4, 4)
    assert len(п.quotes) == 2 and len(set(q.lower().strip(".") for q in п.quotes)) == 2
    assert клиент.calls[0][0].content.count("шла две недели") == 1   # модели повтор не отправляется
