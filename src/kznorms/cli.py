"""CLI: каждый шаг пайплайна — отдельная команда (`uv run kznorms --help`)."""

import typer
from rich import print

app = typer.Typer(help="kz-norms-llm: пайплайн от норм до дообученной модели", no_args_is_help=True)


def _todo(step: str) -> None:
    print(f"[yellow]TODO:[/yellow] шаг [bold]{step}[/bold] ещё не реализован")


@app.command()
def ingest(delay: float = 2.0) -> None:
    """Скачать нормы, распарсить (DOCX/PDF) и нарезать на пункты → data/processed/clauses.jsonl."""
    from kznorms.ingest.download import download_all, load_sources, raw_path
    from kznorms.ingest.parse_docx import parse_docx
    from kznorms.ingest.parse_pdf import parse_pdf
    from kznorms.ingest.segment import segment
    from kznorms.paths import CLAUSES_JSONL, INTERIM

    parsers = {"docx": parse_docx, "pdf": parse_pdf}
    sources = load_sources()
    download_all(sources, delay=delay)

    INTERIM.mkdir(parents=True, exist_ok=True)
    CLAUSES_JSONL.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(CLAUSES_JSONL, "w", encoding="utf-8") as out:
        for src in sources:
            path = raw_path(src)
            if not path.exists():
                continue
            if src.format not in parsers:
                print(f"[yellow]пропуск[/yellow] {src.doc_slug}: парсер {src.format} ещё не готов")
                continue
            blocks = parsers[src.format](path)
            (INTERIM / f"{src.doc_slug}.jsonl").write_text(
                "\n".join(b.model_dump_json() for b in blocks), encoding="utf-8"
            )
            clauses = segment(blocks, src)
            for c in clauses:
                out.write(c.model_dump_json() + "\n")
            total += len(clauses)
            print(f"{src.doc_slug}: {len(blocks)} блоков → {len(clauses)} пунктов")
    print(f"[green]готово[/green]: {total} пунктов → {CLAUSES_JSONL}")


@app.command()
def stats() -> None:
    """Проверка нарезки: пункты на документ, длины, разрывы нумерации, дубли."""
    from collections import defaultdict
    from statistics import median

    from kznorms.ingest.segment import is_successor
    from kznorms.paths import CLAUSES_JSONL
    from kznorms.schemas import Clause

    by_doc: dict[str, list[Clause]] = defaultdict(list)
    for line in CLAUSES_JSONL.read_text(encoding="utf-8").splitlines():
        c = Clause.model_validate_json(line)
        by_doc[c.clause_id.split(":")[0]].append(c)

    for slug, cl in by_doc.items():
        lens = [len(c.text) for c in cl]
        numbered = [c for c in cl if "." in c.number]
        gaps = [
            f"{a.number}→{b.number}"
            for a, b in zip(numbered, numbered[1:], strict=False)
            if not is_successor(a.number, b.number)
        ]
        dups = sum("#" in c.clause_id for c in cl)
        long_ = [c.number for c in cl if len(c.text) > 4000]
        print(f"\n[bold]{slug}[/bold] ({cl[0].doc_code}): {len(cl)} пунктов")
        print(f"  длина текста: медиана {median(lens):.0f}, макс {max(lens)}")
        print(f"  разрывы нумерации ({len(gaps)}): {', '.join(gaps[:10])}")
        print(f"  дубли номеров: {dups}; очень длинные (>4000): {', '.join(long_) or '—'}")


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
def clause(clause_id: str) -> None:
    """Показать один пункт целиком: kznorms clause zhilye-zdaniya:6.2.10"""
    from kznorms.paths import CLAUSES_JSONL
    from kznorms.schemas import Clause

    for line in CLAUSES_JSONL.read_text(encoding="utf-8").splitlines():
        c = Clause.model_validate_json(line)
        if c.clause_id == clause_id:
            typer.echo(f"{c.doc_code}, п. {c.number}  [{c.section}]\n\n{c.text}")
            return
    typer.echo(f"Пункт {clause_id} не найден")


@app.command()
def index(dense: bool = typer.Option(True, help="--no-dense: только BM25, без GPU")) -> None:
    """Построить BM25 и векторный индекс в Qdrant."""
    import time

    from kznorms.config import load_yaml
    from kznorms.paths import BM25_DIR, CHUNKS_JSONL
    from kznorms.retrieval.bm25 import BM25Index
    from kznorms.retrieval.retriever import build_chunks, load_clauses, open_dense, save_chunks

    cfg = load_yaml("retrieval")
    clauses = load_clauses()
    chunks = build_chunks(cfg, clauses)
    save_chunks(chunks)
    n_split = len({c.clause_id for c in chunks if not c.chunk_id.endswith("#0")})
    print(f"{len(clauses)} пунктов → {len(chunks)} фрагментов ({n_split} пунктов разрезано)")
    print(f"  → {CHUNKS_JSONL}")

    t = time.perf_counter()
    BM25Index.build(chunks).save(BM25_DIR)
    print(f"BM25: {time.perf_counter() - t:.1f} с → {BM25_DIR}")

    if dense:
        t = time.perf_counter()
        idx = open_dense(cfg)
        texts = [c.text for c in chunks]
        trunc = idx.embedder.count_truncated(texts)
        if trunc:
            print(f"[yellow]{trunc} фрагментов длиннее {cfg['max_seq_length']} токенов[/yellow]")
        idx.build(chunks, batch_size=cfg["batch_size"])
        idx.close()
        print(f"Qdrant ({cfg['embedder']}): {time.perf_counter() - t:.1f} с")
    print("[green]готово[/green]")


@app.command()
def search(
    query: str,
    mode: str = typer.Option(None, help="bm25 | dense | hybrid | hybrid_rerank"),
    k: int = typer.Option(None, help="сколько пунктов показать (по умолчанию final_k)"),
    full: bool = typer.Option(False, help="показать фрагмент целиком"),
) -> None:
    """Поиск пунктов: kznorms search "ширина марша лестницы" --mode bm25"""
    from kznorms.retrieval.retriever import Retriever

    r = Retriever()
    r.warmup(mode)
    hits = r.retrieve(query, k=k, mode=mode)
    for i, h in enumerate(hits, 1):
        body = h.chunk.split("\n", 1)[-1]
        body = body if full else body[:300] + ("…" if len(body) > 300 else "")
        print(f"\n[bold]{i}. {h.clause.clause_id}[/bold]  score={h.score:.4f}")
        print(f"   [dim]{h.clause.doc_code} · {h.clause.section}[/dim]")
        typer.echo(f"   {body}")
    t = ", ".join(f"{name} {ms:.0f} мс" for name, ms in r.timings_ms.items())
    print(f"\n[dim]режим {mode or r.cfg['mode']}: {t}[/dim]")


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
