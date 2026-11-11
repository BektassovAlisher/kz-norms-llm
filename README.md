# kz-norms-llm

Локальная LLM (Qwen3-4B + QLoRA через Unsloth), которая отвечает на вопросы по строительным
нормам РК строго по найденным пунктам: со ссылкой, дословной цитатой и JSON-ответом,
а если ответа в нормах нет — честно говорит `not_found`.

> Статус: каркас проекта. Метрики и демо появятся по мере работы.

## Быстрый старт

Рекомендуется WSL2 (Ubuntu): Unsloth и llama.cpp там ставятся без проблем.

```bash
# uv: https://docs.astral.sh/uv/
uv sync                 # базовые зависимости + dev
uv sync --extra train   # + Unsloth (только Linux/WSL2 с NVIDIA GPU)
cp .env.example .env    # и заполнить ключи
uv run kznorms --help
uv run pytest
```

## Структура

```
configs/        YAML-конфиги поиска, синтетики, обучения
data/           sources.csv (список норм), eval/ (ручная разметка); остальное генерируется
src/kznorms/
  ingest/       скачивание, парсинг, нарезка на пункты
  retrieval/    BM25 + BGE-M3 + Qdrant + реранкер
  rag/          промпт, LLM-клиент, проверка цитат, пайплайн
  synth/        синтетический датасет (учитель → фильтр)
  train/        QLoRA (Unsloth), эмбеддинги, экспорт GGUF
  eval/         метрики поиска и генерации, отчёт
  agent/        этап 2: проверка фрагментов проекта
  api/          FastAPI
ui/             Gradio-демо
```

## Пайплайн

```
make ingest → make index → make synth → make train-llm → make export → make eval → make serve
```
