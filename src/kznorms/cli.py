"""CLI: каждый шаг пайплайна — отдельная команда (`uv run kznorms --help`)."""

import typer
from rich import print

app = typer.Typer(help="kz-norms-llm: пайплайн от норм до дообученной модели", no_args_is_help=True)


def _todo(step: str) -> None:
    print(f"[yellow]TODO:[/yellow] шаг [bold]{step}[/bold] ещё не реализован")


@app.command()
def ingest(delay: float = 2.0) -> None:
    """Скачать нормы из sources.csv и распарсить в блоки (data/interim/)."""
    from kznorms.ingest.download import download_all, load_sources, raw_path
    from kznorms.ingest.parse_docx import parse_docx
    from kznorms.paths import INTERIM

    sources = load_sources()
    download_all(sources, delay=delay)

    INTERIM.mkdir(parents=True, exist_ok=True)
    for src in sources:
        path = raw_path(src)
        if not path.exists():
            continue
        if src.format != "docx":
            print(f"[yellow]пропуск[/yellow] {src.doc_slug}: парсер {src.format} ещё не готов")
            continue
        blocks = parse_docx(path)
        out = INTERIM / f"{src.doc_slug}.jsonl"
        out.write_text("\n".join(b.model_dump_json() for b in blocks), encoding="utf-8")
        n_tables = sum(b.type == "table" for b in blocks)
        print(f"{src.doc_slug}: {len(blocks)} блоков, из них таблиц: {n_tables}")


@app.command()
def peek(slug: str, n: int = 40, start: int = 0) -> None:
    """Показать блоки документа из data/interim/ — проверить парсинг глазами."""
    from kznorms.paths import INTERIM
    from kznorms.schemas import Block

    lines = (INTERIM / f"{slug}.jsonl").read_text(encoding="utf-8").splitlines()
    for line in lines[start : start + n]:
        b = Block.model_validate_json(line)
        typer.echo(f"[{b.type}] {b.text[:150]}")


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
