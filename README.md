# LLM-агент для анализа отзывов покупателей

Выпускная квалификационная работа: агент анализирует отзывы на товар продавца маркетплейса и на товары конкурентов и формирует рекомендации по развитию товара.

Документы: [ТЗ](docs/tz.md), [архитектура](docs/architecture.md), [обзор аналогов](docs/analogs.md), [план](docs/roadmap.md).

## Что уже работает

| Часть | Где | Требования ТЗ |
|---|---|---|
| Загрузка отзывов из CSV/XLSX | `src/review_agent/ingest/` | FR-1 |
| Очистка и разбиение на предложения | `src/review_agent/preprocess/` | FR-2, FR-3 |
| LLM-шлюз: проверка формата, повторы, кэш, журнал, стоимость | `src/review_agent/llm/` | NFR-3 – NFR-6 |
| Извлечение аспектов и тональности (шаг 4 конвейера) | `src/review_agent/pipeline/extract.py` | FR-5, FR-6 |
| Агрегация: охват, доля негатива, динамика по месяцам (шаг 5) | `src/review_agent/pipeline/aggregate.py` | FR-8, FR-9 |
| Сравнение с конкурентами, расчётная часть (шаг 7) | `src/review_agent/pipeline/compare.py` | FR-12 |
| Приоритизация по формуле (шаг 10) | `src/review_agent/pipeline/prioritize.py` | FR-16 |
| Проверка опоры на отзывы (шаг 11) | `src/review_agent/pipeline/grounding.py` | FR-17 |
| Хранилище SQLite | `src/review_agent/storage/` | 4.3 |
| Сравнение моделей на размеченной выборке | `eval/` | раздел 7.2 |

Клиенты GigaChat, YandexGPT и Qwen проверены только на заглушке HTTP. С настоящими сервисами они ещё не запускались: нужны ключи.

Ещё не сделано: справочник аспектов (шаг 3), выделение проблем (шаг 6), генерация рекомендаций и критик (шаги 8–9), оркестратор, отчёт, веб-интерфейс.

## Установка

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -e ".[dev]"
```

## Тесты

```bash
pytest
```

## Мини-эксперимент по выбору LLM

Порядок действий описан в [eval/README.md](eval/README.md): разметить выборку, вписать ключи в `.env`, запустить сравнение.

```bash
python -m eval.run_experiment --models gigachat yandex qwen
```
