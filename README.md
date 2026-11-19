# LLM-агент для анализа отзывов покупателей

ВКР: разработка LLM-агента, который анализирует отзывы покупателей на товар продавца и
конкурентов и формирует обоснованные рекомендации по развитию товара.

Документы: `docs/tz.md` (ТЗ), `docs/architecture.md` (архитектура), `docs/roadmap.md` (план).

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Что уже работает (без ключей API)

| Часть | Где | Шаг конвейера |
|---|---|---|
| Загрузка CSV/XLSX, сопоставление колонок | `src/review_agent/ingest/` | 1 |
| Очистка, предложения, дубликаты, маскирование персональных данных | `src/review_agent/preprocess/` | 2 |
| Извлечение аспектов с проверкой ответа модели | `src/review_agent/pipeline/extract.py` | 4 |
| Агрегация: охват, доля негатива, динамика | `src/review_agent/pipeline/aggregate.py` | 5 |
| Приоритизация по формуле | `src/review_agent/pipeline/prioritize.py` | 10 |
| Проверка опоры на отзывы | `src/review_agent/pipeline/grounding.py` | 11 |
| LLM-шлюз: схема, повторы, кэш, журнал, стоимость | `src/review_agent/llm/gateway.py` | — |
| Клиенты GigaChat, YandexGPT, Ollama, заглушка | `src/review_agent/llm/` | — |
| SQLite | `src/review_agent/storage/db.py` | — |
| Метрики качества, словарный метод для сравнения | `src/review_agent/evaluation/` | — |

Клиенты GigaChat и YandexGPT проверены только на подменённой сети: на реальном API их нужно
проверить, когда появятся ключи.

## Данные и мини-эксперимент

```bash
python scripts/prepare_rureviews.py                       # выборки для разметки
python scripts/run_extraction_eval.py --provider baseline  # словарный метод
python scripts/run_extraction_eval.py --provider fake      # пробный прогон без ключей
```

Как размечать: `eval/annotation_guide.md`. Про данные и персональные данные: `data/README.md`.

## Ключи API

Только через переменные окружения, в репозиторий не попадают:

| Провайдер | Переменные |
|---|---|
| GigaChat | `GIGACHAT_AUTH_KEY` (ключ авторизации), при необходимости `GIGACHAT_CA_BUNDLE` (сертификат НУЦ Минцифры) |
| YandexGPT | `YANDEX_API_KEY`, `YANDEX_FOLDER_ID` |
| Ollama | `OLLAMA_URL` (по умолчанию `http://localhost:11434`) |

```bash
python scripts/run_extraction_eval.py --provider gigachat --model GigaChat-2-Pro
```
