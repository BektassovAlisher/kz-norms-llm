"""CLI: каждый шаг пайплайна — отдельная команда (`uv run kznorms --help`)."""

import typer
from rich import print

app = typer.Typer(help="kz-norms-llm: пайплайн от норм до дообученной модели", no_args_is_help=True)


def _todo(step: str) -> None:
    print(f"[yellow]TODO:[/yellow] шаг [bold]{step}[/bold] ещё не реализован")


@app.command()
def ingest() -> None:
    """Скачать нормы из sources.csv, распарсить и порезать на пункты."""
    _todo("ingest")


@app.command()
def stats() -> None:
    """Статистика по clauses.jsonl (пункты на документ, длины, мусор)."""
    _todo("stats")


@app.command()
def index() -> None:
    """Построить BM25 и векторный индекс в Qdrant."""
    _todo("index")


@app.command()
def synth() -> None:
    """Сгенерировать и отфильтровать синтетический SFT-датасет."""
    _todo("synth")


@app.command("train-llm")
def train_llm(config: str = "sft_qwen3_4b") -> None:
    """QLoRA-дообучение Qwen3-4B через Unsloth."""
    _todo(f"train-llm ({config})")


@app.command("train-emb")
def train_emb(config: str = "emb_bge_m3") -> None:
    """Дообучение эмбеддингов BGE-M3."""
    _todo(f"train-emb ({config})")


@app.command()
def export() -> None:
    """Слить LoRA и выгрузить GGUF."""
    _todo("export")


@app.command("eval")
def eval_() -> None:
    """Посчитать метрики поиска и генерации → reports/results.md."""
    _todo("eval")


@app.command()
def ask(question: str, model: str = "base") -> None:
    """Задать вопрос через RAG-пайплайн."""
    _todo(f"ask ({model}): {question}")


if __name__ == "__main__":
    app()
