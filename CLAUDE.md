# kz-norms-llm — инструкции для Claude

Полный контекст проекта, принятые решения и план: **docs/HANDOFF.md** — прочитай его в начале работы.

## Как работать с владельцем проекта
- Общение на русском, на «ты». Владелец учится: на каждом шаге объясняй теорию
  (почему так, какие ловушки), потом код. Не просто делай — объясняй.
- Коммить и пушь сразу в `main`. Автор коммитов — только владелец (его локальный git config).
  Никаких `Co-Authored-By`, упоминаний Claude или ссылок на сессии в коммитах, PR и коде.
- Окружение: Windows (без WSL пока), uv, VS Code. `make` нет — команды запускать
  через `uv run kznorms <команда>`. Для обучения (Unsloth) позже переходим в WSL2.
- Железо: RTX 3060 Ti 8 ГБ, Ryzen 5 5600X, 16 ГБ RAM — всё планировать под 8 ГБ VRAM.

## Команды
```
uv sync                       # зависимости
uv run pytest                 # тесты (должны быть зелёными перед коммитом)
uv run ruff check . && uv run ruff format .
uv run kznorms ingest         # скачать нормы из sources.csv, data/raw → data/processed/clauses.jsonl
uv run kznorms stats [--gaps]  # качество нарезки по документам (покрытие, разрывы, гиганты)
uv run kznorms clause <id>    # показать пункт, например zhilye-zdaniya:6.2.10
uv run kznorms peek <slug>    # блоки после парсинга
uv run kznorms index          # чанки + BM25 + BGE-M3 → Qdrant (artifacts/), --no-dense без GPU
uv run kznorms search "<вопрос>" --mode bm25|dense|hybrid|hybrid_rerank --k 5
uv run kznorms qa-check       # проверка разметки data/eval/qa_test.jsonl
uv run kznorms qa-review      # проверить черновые вопросы руками
uv run kznorms eval-retrieval [--drafts]   # метрики поиска → reports/retrieval.md
```
