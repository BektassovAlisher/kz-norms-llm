.PHONY: setup setup-train ingest stats index synth train-llm train-emb export eval serve test lint

setup:          ## базовое окружение
	uv sync

setup-train:    ## + Unsloth (только WSL2/Linux с GPU)
	uv sync --extra train

ingest:         ## скачать и распарсить нормы → data/processed/clauses.jsonl
	uv run kznorms ingest

stats:
	uv run kznorms stats

index:          ## BM25 + эмбеддинги в Qdrant
	uv run kznorms index

synth:          ## синтетический датасет для SFT
	uv run kznorms synth

train-llm:      ## QLoRA Qwen3-4B (перед этим: docker compose down)
	uv run kznorms train-llm

train-emb:
	uv run kznorms train-emb

export:
	uv run kznorms export

eval:
	uv run kznorms eval

serve:
	docker compose up -d

test:
	uv run pytest

lint:
	uv run ruff check . && uv run ruff format --check .
