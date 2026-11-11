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


@app.command("qa-check")
def qa_check() -> None:
    """Проверить разметку data/eval/qa_test.jsonl: id пунктов, дубли, баланс."""
    from collections import Counter

    from kznorms.eval.eval_retrieval import QA_TEST, load_items
    from kznorms.retrieval.retriever import load_clauses

    items = load_items()
    known = {c.clause_id for c in load_clauses()}
    errors = []
    for id_, n in Counter(it.id for it in items).items():
        if n > 1:
            errors.append(f"{id_}: id повторяется {n} раз")
    for q, n in Counter(it.question.strip().lower() for it in items).items():
        if n > 1:
            errors.append(f"вопрос повторяется {n} раз: {q}")
    for it in items:
        if it.answerable and not it.gold_clause_ids:
            errors.append(f"{it.id}: answerable, но нет gold_clause_ids")
        if not it.answerable and it.gold_clause_ids:
            errors.append(f"{it.id}: без ответа, но указаны gold_clause_ids")
        for g in it.gold_clause_ids:
            if g not in known:
                errors.append(f"{it.id}: пункта {g} нет в clauses.jsonl (перенарезка?)")

    ans = [it for it in items if it.answerable]
    docs = Counter(g.split(":")[0] for it in ans for g in it.gold_clause_ids[:1])
    print(f"{QA_TEST}: {len(items)} вопросов, проверено руками: {sum(it.verified for it in items)}")
    print(
        f"  с ответом {len(ans)}, без ответа {len(items) - len(ans)} "
        f"({(len(items) - len(ans)) / max(len(items), 1):.0%}, цель ≈25%)"
    )
    print(f"  по документам: {dict(docs)}")
    print(f"  по типам: {dict(Counter(it.type for it in items))}")
    for e in errors:
        print(f"[red]ошибка[/red] {e}")
    if errors:
        raise typer.Exit(1)
    print("[green]разметка в порядке[/green]")


@app.command("qa-review")
def qa_review(all_: bool = typer.Option(False, "--all", help="включая уже проверенные")) -> None:
    """Пройти черновые вопросы руками: gold-пункты + что нашёл поиск → подтвердить/поправить."""
    from kznorms.eval.eval_retrieval import load_items, save_items
    from kznorms.retrieval.retriever import Retriever

    items = load_items()
    todo = [it for it in items if all_ or not it.verified]
    if not todo:
        print("[green]все вопросы проверены[/green]")
        return
    r = Retriever()
    r.warmup("hybrid_rerank")
    help_ = (
        "[y] верно  [g] заменить gold  [a] заменить ответ  [u] без ответа  "
        "[d] удалить  [s] пропустить  [q] выйти"
    )
    for n, it in enumerate(todo, 1):
        print(f"\n[bold]━━ {n}/{len(todo)} · {it.id} · {it.type}[/bold]")
        print(f"[bold cyan]{it.question}[/bold cyan]")
        if it.note:
            print(f"[dim]заметка: {it.note}[/dim]")
        if it.answerable:
            print(f"эталон: {it.gold_answer}")
        for g in it.gold_clause_ids:
            c = r.clauses.get(g)
            typer.echo(f"\n  GOLD {g}\n  {c.text if c else '!!! пункта нет в корпусе'}")
        print("\n  [dim]поиск hybrid_rerank (★ = gold):[/dim]")
        for h in r.retrieve(it.question, k=5):
            mark = "★" if h.clause.clause_id in it.gold_clause_ids else " "
            body = h.chunk.split("\n", 1)[-1][:220].replace("\n", " ")
            typer.echo(f"  {mark} {h.score:.3f} {h.clause.clause_id}: {body}")
        while True:
            cmd = typer.prompt(help_, default="s").strip().lower()
            if cmd == "y":
                it.verified = True
            elif cmd == "g":
                ids = typer.prompt("gold через пробел").split()
                if unknown := [g for g in ids if g not in r.clauses]:
                    print(f"[red]нет таких пунктов: {unknown}[/red]")
                    continue
                it.gold_clause_ids, it.answerable = ids, True
                continue  # покажем меню снова: можно сразу подтвердить
            elif cmd == "a":
                it.gold_answer = typer.prompt("эталонный ответ")
                continue
            elif cmd == "u":
                it.answerable, it.gold_clause_ids, it.gold_answer = False, [], ""
                continue
            elif cmd == "d":
                items.remove(it)
            elif cmd == "q":
                save_items(items)
                return
            elif cmd != "s":
                continue
            break
        save_items(items)  # после каждого вопроса — ничего не потеряется при выходе
    print("[green]готово[/green]")


@app.command("eval-retrieval")
def eval_retrieval(
    drafts: bool = typer.Option(False, help="учитывать и непроверенные вопросы"),
    modes: str = typer.Option("bm25,dense,hybrid,hybrid_rerank", help="через запятую"),
) -> None:
    """Hit@5, Recall@5, MRR@10 по режимам → reports/retrieval.md."""
    from rich.table import Table

    from kznorms.eval.eval_retrieval import evaluate, load_items, to_markdown, write_misses
    from kznorms.paths import REPORTS
    from kznorms.retrieval.retriever import Retriever

    items = [it for it in load_items() if drafts or it.verified]
    if not items:
        print(
            "[yellow]нет проверенных вопросов[/yellow]: пройди `kznorms qa-review` "
            "или запусти с --drafts"
        )
        raise typer.Exit(1)
    mode_list = [m.strip() for m in modes.split(",")]
    r = Retriever()
    r.warmup("hybrid_rerank" if "hybrid_rerank" in mode_list else mode_list[-1])
    results = [evaluate(r, items, m) for m in mode_list]

    n_ans = sum(it.answerable for it in items)
    table = Table(title=f"Поиск: {n_ans} вопросов с ответом" + (" (черновик)" if drafts else ""))
    for col in ("режим", "Hit@5", "Recall@5", "MRR@10", "p50, мс"):
        table.add_column(col, justify="right")
    for res in results:
        table.add_row(
            res.mode,
            f"{res.hit5:.3f}",
            f"{res.recall5:.3f}",
            f"{res.mrr10:.3f}",
            f"{res.latency_ms_p50:.0f}",
        )
    print(table)

    REPORTS.mkdir(parents=True, exist_ok=True)
    note = "ЧЕРНОВИК: в выборке есть непроверенные вопросы." if drafts else "Только verified."
    (REPORTS / "retrieval.md").write_text(
        to_markdown(results, n_ans, len(items) - n_ans, note), encoding="utf-8"
    )
    write_misses(results, REPORTS / "retrieval_misses.json")
    print(f"→ {REPORTS / 'retrieval.md'}")


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
