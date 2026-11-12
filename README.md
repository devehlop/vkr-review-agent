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
| Справочник аспектов по выборке отзывов (шаг 3) | `src/review_agent/pipeline/taxonomy.py` | FR-7 |
| Агрегация: охват, доля негатива, динамика по месяцам (шаг 5) | `src/review_agent/pipeline/aggregate.py` | FR-8, FR-9 |
| Выделение проблем: кластеризация и название от модели (шаг 6) | `src/review_agent/pipeline/issues.py` | FR-10, FR-11 |
| Сравнение с конкурентами, расчётная часть (шаг 7) | `src/review_agent/pipeline/compare.py` | FR-12 |
| Генерация рекомендаций и цикл «генератор — критик» (шаги 8–9) | `src/review_agent/pipeline/recommend.py`, `critic.py` | FR-13 – FR-15 |
| Приоритизация по формуле (шаг 10) | `src/review_agent/pipeline/prioritize.py` | FR-16 |
| Проверка опоры на отзывы (шаг 11) | `src/review_agent/pipeline/grounding.py` | FR-17 |
| Оркестратор: полный прогон от отзывов к рекомендациям | `src/review_agent/pipeline/orchestrator.py` | 3.4 |
| Отчёт в Markdown и запуск из командной строки | `src/review_agent/report/`, `src/review_agent/cli.py` | FR-18 |
| Хранилище SQLite | `src/review_agent/storage/` | 4.3 |
| Сравнение моделей на размеченной выборке | `eval/` | раздел 7.2 |

Весь конвейер проверен только на заглушке модели, которая отвечает по простым правилам. С настоящими GigaChat, YandexGPT и Qwen он ещё не запускался: нужны ключи. Качество промптов и рекомендаций поэтому пока не оценено.

Ещё не сделано: веб-интерфейс, рекомендации по сильным сторонам, LLM-резюме сравнения с конкурентами, динамика в отчёте, отметка подозрительных отзывов, нейросетевые эмбеддинги, агент вопросов, экспорт в DOCX.

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

## Запуск анализа

```bash
python -m review_agent.cli --own отзывы.xlsx --product "Платье летнее" --competitor "Платье конкурента=конкурент.csv" --model gigachat --out report.md
```

Ключи моделей задаются в `.env` (образец: `.env.example`). Повторный запуск берёт ответы модели из кэша `.cache/`.

## Мини-эксперимент по выбору LLM

Порядок действий описан в [eval/README.md](eval/README.md): разметить выборку, вписать ключи в `.env`, запустить сравнение.

```bash
python -m eval.run_experiment --models gigachat yandex qwen
```
