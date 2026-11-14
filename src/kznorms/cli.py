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
def stats(gaps: bool = typer.Option(False, help="перечислить разрывы нумерации")) -> None:
    """Качество нарезки по документам: покрытие текста, разрывы нумерации, дубли, гиганты.

    Покрытие — доля текста после парсинга (data/interim), попавшая в пункты. Низкое
    покрытие значит, что текст где-то теряется (например, ушёл в пропускаемый раздел).
    «строчн.» — пункты, текст которых начинается со строчной: почти всегда это ложный номер
    («…по Таблице» + «В.8 как для плит») или сдвиг нумерации.
    """
    from collections import defaultdict
    from statistics import median

    from rich.table import Table

    from kznorms.ingest.segment import is_successor
    from kznorms.paths import CLAUSES_JSONL, INTERIM
    from kznorms.schemas import Block, Clause

    by_doc: dict[str, list[Clause]] = defaultdict(list)
    for line in CLAUSES_JSONL.read_text(encoding="utf-8").splitlines():
        c = Clause.model_validate_json(line)
        by_doc[c.clause_id.split(":")[0]].append(c)

    table = Table(
        title=f"{sum(map(len, by_doc.values()))} пунктов в {len(by_doc)} документах",
        caption="покр — доля текста в пунктах; строч — пункты со строчной буквы; "
        "мед/макс — длина текста пункта",
    )
    table.add_column("документ", no_wrap=True)
    for col in ("пункт", "покр", "разр", "дубл", "строч", ">4k", "мед", "макс"):
        table.add_column(col, justify="right")
    gap_lines = []
    for slug, cl in by_doc.items():
        interim = INTERIM / f"{slug}.jsonl"
        total = sum(
            len(Block.model_validate_json(x).text)
            for x in interim.read_text(encoding="utf-8").splitlines()
            if x.strip()
        ) if interim.exists() else 0  # fmt: skip
        got = sum(len(c.text) + len(c.number) + 1 for c in cl)
        nums = [c.number for c in cl if "." in c.number]
        doc_gaps = [
            f"{a}→{b}" for a, b in zip(nums, nums[1:], strict=False) if not is_successor(a, b)
        ]
        lens = [len(c.text) for c in cl]
        table.add_row(
            slug,
            str(len(cl)),
            f"{got / total:.0%}" if total else "—",
            str(len(doc_gaps)),
            str(sum("#" in c.clause_id for c in cl)),
            str(sum(c.text[:1].islower() for c in cl)),
            str(sum(x > 4000 for x in lens)),
            f"{median(lens):.0f}",
            str(max(lens)),
        )
        if doc_gaps:
            gap_lines.append(f"[bold]{slug}[/bold]: {', '.join(doc_gaps[:12])}")
    print(table)
    if gaps:
        print("\n".join(gap_lines))
    else:
        print("[dim]разрывы вида 1.3→3.1 — пропускаемые «Нормативные ссылки»; список: --gaps[/dim]")


@app.command()
def splits() -> None:
    """Сплит документов train/val/test: группы, доли пунктов и текста, вопросы теста.

    Долю считаем по пунктам и по тексту, а не по числу документов: документы отличаются
    по объёму в 10 раз (СН по шуму — 122 пункта, СП по высотным — 862).
    """
    from collections import Counter, defaultdict

    from rich.table import Table

    from kznorms.eval.eval_retrieval import load_items
    from kznorms.ingest.download import load_sources
    from kznorms.retrieval.retriever import load_clauses
    from kznorms.splits import SPLITS, check_sources, doc_of, item_split

    sources = load_sources()
    if errors := check_sources(sources):
        for e in errors:
            print(f"[red]ошибка[/red] {e}")
        raise typer.Exit(1)
    split_of = {s.doc_slug: s.split for s in sources}
    n_clauses, n_chars = Counter(), Counter()
    for c in load_clauses():
        n_clauses[split_of[doc_of(c.clause_id)]] += 1
        n_chars[split_of[doc_of(c.clause_id)]] += len(c.text)
    groups: dict[str, set[str]] = defaultdict(set)
    for s in sources:
        groups[s.split].add(s.group)
    qa = Counter(item_split(it, split_of) for it in load_items() if it.answerable)

    table = Table(title="Сплит по группам документов (СН+СП одной темы — вместе)")
    for col in ("сплит", "групп", "докум.", "пунктов", "доля пунктов", "доля текста", "QA"):
        table.add_column(col, justify="right")
    for sp in SPLITS:
        table.add_row(
            sp,
            str(len(groups[sp])),
            str(sum(s.split == sp for s in sources)),
            str(n_clauses[sp]),
            f"{n_clauses[sp] / n_clauses.total():.0%}",
            f"{n_chars[sp] / n_chars.total():.0%}",
            str(qa[sp]),
        )
    print(table)
    for sp in SPLITS:
        print(f"[bold]{sp}[/bold]: {', '.join(sorted(groups[sp]))}")
    if qa["mixed"]:
        print(f"[yellow]{qa['mixed']} вопр. с gold из разных сплитов[/yellow] — см. qa-check")


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
    from kznorms.splits import item_split, load_splits

    items = load_items()
    split_of = load_splits()
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
    by_split = Counter(item_split(it, split_of) for it in ans)
    print(f"  с ответом по сплитам: {dict(by_split)}")
    for it in ans:
        sp = item_split(it, split_of)
        if sp != "test":
            print(
                f"[yellow]внимание[/yellow] {it.id}: gold в {sp}-документах "
                f"{it.gold_clause_ids} — в итоговые метрики не войдёт"
            )
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
    all_splits: bool = typer.Option(
        False, "--all-splits", help="и вопросы с gold в train/val-документах (не для итогов)"
    ),
) -> None:
    """Hit@5, Recall@5, MRR@10 по режимам → reports/retrieval.md."""
    from rich.table import Table

    from kznorms.eval.eval_retrieval import evaluate, load_items, to_markdown, write_misses
    from kznorms.paths import REPORTS
    from kznorms.retrieval.retriever import Retriever
    from kznorms.splits import item_split, load_splits

    split_of = load_splits()
    items = [it for it in load_items() if drafts or it.verified]
    if not all_splits:
        before = len(items)
        items = [it for it in items if item_split(it, split_of) == "test"]
        if skipped := before - len(items):
            print(f"[dim]пропущено {skipped} вопр. не из test-документов (--all-splits)[/dim]")
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
    note += " Все сплиты." if all_splits else " Только test-документы."
    (REPORTS / "retrieval.md").write_text(
        to_markdown(results, n_ans, len(items) - n_ans, note), encoding="utf-8"
    )
    write_misses(results, REPORTS / "retrieval_misses.json")
    print(f"→ {REPORTS / 'retrieval.md'}")


@app.command()
def synth(
    teacher: str = typer.Option(None, help="local (llama-server) | big (API из .env)"),
    n_clauses: int = typer.Option(None, help="сколько пунктов взять (по умолчанию из конфига)"),
    stage: str = typer.Option(
        "all", help="all | questions | contexts | answers (локальный учитель: по одному)"
    ),
) -> None:
    """Синтетика: пункты train/val → вопросы → контексты → ответы учителя → фильтр → data/sft/.

    С локальным учителем этапы запускаются по одному: учитель и модели поиска вместе не
    помещаются в 8 ГБ. questions и answers — с запущенным llama-server, contexts — без него.
    """
    import json

    from kznorms.config import load_yaml
    from kznorms.paths import SFT, SYNTHETIC
    from kznorms.retrieval.retriever import Retriever
    from kznorms.splits import load_splits
    from kznorms.synth.run import STAGES, run
    from kznorms.synth.teacher import QuotaExhausted, get_teacher

    stages = STAGES if stage == "all" else (stage,)
    if any(s not in STAGES for s in stages):
        raise typer.BadParameter(f"--stage: all | {' | '.join(STAGES)}")
    cfg = load_yaml("synth")
    cfg["n_clauses"] = n_clauses or cfg["n_clauses"]
    name = teacher or cfg["teacher"]
    needs_teacher = stages != ("contexts",)
    t = get_teacher(name, cfg) if needs_teacher else None
    if t:
        print(f"учитель: [bold]{t.model}[/bold]")
    try:
        rep = run(t, Retriever(), load_splits(), cfg, stages)
    except QuotaExhausted as e:
        print(f"[red]учитель перестал отвечать[/red] (квота или ключ): {e}")
        print("Готовые ответы в кэше — перезапусти позже, продолжится с места.")
        raise typer.Exit(1) from e
    if rep:
        typer.echo(json.dumps(rep, ensure_ascii=False, indent=1))
        print(f"[green]готово[/green] → {SFT}/train.jsonl, val.jsonl; разбор — {SYNTHETIC}")
    else:
        print(f"[green]этап {stage} готов[/green] → {SYNTHETIC}")


@app.command("train-llm")
def train_llm(
    config: str = typer.Option("sft_qwen3_4b", help="конфиг из configs/"),
    n_train: int = typer.Option(0, help="первые N примеров (абляции 500/1000/2000/4000); 0 — все"),
    max_steps: int = typer.Option(0, help="ограничить число шагов (пробный запуск); 0 — эпохи"),
    run_name: str = typer.Option(None, help="имя прогона → artifacts/qwen3-4b-lora/<имя>"),
    r: int = typer.Option(0, help="ранг LoRA вместо конфига (абляция r 8/16/32)"),
) -> None:
    """QLoRA-дообучение Qwen3-4B через Unsloth (только WSL2/Linux) → адаптер + summary.json."""
    import json

    from kznorms.config import load_yaml
    from kznorms.train.sft_unsloth import train

    cfg = load_yaml(config)
    if r:
        cfg["lora"]["r"] = cfg["lora"]["alpha"] = r
    summary = train(cfg, n_train or None, max_steps or None, run_name)
    summary.pop("config")
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=1))


@app.command("train-emb")
def train_emb(config: str = "emb_bge_m3") -> None:
    """Дообучение эмбеддингов BGE-M3."""
    _todo(f"train-emb ({config})")


@app.command()
def export() -> None:
    """Слить LoRA и выгрузить GGUF."""
    _todo("export")


@app.command("eval-generation")
def eval_generation(
    model: str = typer.Option("base", help="base | big"),
    modes: str = typer.Option("oracle,e2e", help="через запятую"),
    drafts: bool = typer.Option(False, help="учитывать и непроверенные вопросы"),
    all_splits: bool = typer.Option(False, "--all-splits", help="и вопросы не из test"),
    limit: int = typer.Option(0, help="только первые N вопросов (быстрая проверка)"),
    reverify: bool = typer.Option(
        False, help="не звать LLM: пересчитать verify и метрики по сохранённому .jsonl"
    ),
) -> None:
    """Ответы LLM по тестовому набору (oracle / e2e) → reports/generation_<model>.md."""
    from rich.progress import track
    from rich.table import Table

    from kznorms.eval.eval_generation import (
        LABELS,
        _fmt,
        load_records,
        run_item,
        save_records,
        summarize,
        to_markdown,
    )
    from kznorms.eval.eval_generation import (
        reverify as reverify_records,
    )
    from kznorms.eval.eval_retrieval import load_items
    from kznorms.paths import REPORTS
    from kznorms.rag.llm_client import get_llm
    from kznorms.rag.pipeline import RagPipeline
    from kznorms.splits import item_split, load_splits

    jsonl = REPORTS / f"generation_{model}.jsonl"
    mode_list = [m.strip() for m in modes.split(",")]
    if reverify:
        records = [r for r in load_records(jsonl) if r.mode in mode_list]
        reverify_records(records)
    else:
        split_of = load_splits()
        items = [it for it in load_items() if drafts or it.verified]
        if not all_splits:
            items = [it for it in items if item_split(it, split_of) == "test"]
        items = items[:limit] if limit else items
        if not items:
            print("[yellow]нет вопросов[/yellow]: пройди `kznorms qa-review` или --drafts")
            raise typer.Exit(1)
        pipe = RagPipeline(get_llm(model))
        pipe.retriever.warmup("hybrid_rerank")
        k = pipe.retriever.cfg["final_k"]
        records = [
            run_item(pipe, it, m, k)
            for m in mode_list
            for it in track(items, description=f"{model}: {m}")
        ]
    summaries = [summarize(records, m) for m in mode_list]

    table = Table(title=f"Генерация {model}: {summaries[0].n_ans} с ответом, "
                  f"{summaries[0].n_unans} без" + (" (черновик)" if drafts else ""))  # fmt: skip
    table.add_column("метрика")
    for s in summaries:
        table.add_column(s.mode, justify="right")
    for key, label in LABELS.items():
        table.add_row(label, *(_fmt(s.metrics.get(key), key) for s in summaries))
    print(table)

    REPORTS.mkdir(parents=True, exist_ok=True)
    note = "ЧЕРНОВИК: есть непроверенные вопросы." if drafts else "Только verified."
    note += " Все сплиты." if all_splits else " Только test-документы."
    md = REPORTS / f"generation_{model}.md"
    md.write_text(to_markdown(summaries, records, model, note), encoding="utf-8")
    save_records(records, jsonl)
    print(f"→ {md} (+ .jsonl — все ответы для разбора)")


@app.command()
def ui(
    model: str = typer.Option("base", help="base | big"),
    port: int = typer.Option(7860, help="порт веб-интерфейса"),
) -> None:
    """Веб-интерфейс Gradio: http://127.0.0.1:7860 (нужен запущенный llama-server)."""
    from kznorms.api.ui import CSS, build_app

    def factory():
        from kznorms.rag.llm_client import get_llm
        from kznorms.rag.pipeline import RagPipeline

        pipe = RagPipeline(get_llm(model))
        pipe.retriever.warmup("hybrid_rerank")
        return pipe

    # модели поиска грузим сразу: иначе первый вопрос пользователя ждёт их загрузку ~30 с
    print("загружаю поиск (BGE-M3, реранкер)…")
    pipe = factory()
    build_app(lambda: pipe).launch(server_name="127.0.0.1", server_port=port, css=CSS)


@app.command("eval")
def eval_() -> None:
    """Посчитать метрики поиска и генерации → reports/results.md."""
    _todo("eval")


@app.command()
def ask(question: str, model: str = "base") -> None:
    """Задать вопрос через RAG: kznorms ask "ширина марша лестницы в жилом доме" """
    from kznorms.rag.llm_client import get_llm
    from kznorms.rag.pipeline import RagPipeline

    resp = RagPipeline(get_llm(model)).answer(question)
    a = resp.answer
    mark = "[green]verify ok[/green]" if resp.verified else "[red]verify не пройден[/red]"
    print(f"[bold]{a.status}[/bold]  {mark}")
    typer.echo(f"\n{a.answer}\n")
    for c in a.citations:
        typer.echo(f"  [{c.clause_id}] «{c.quote}»")  # echo, не rich: [..] — не разметка
    for p in resp.problems:
        typer.echo(f"  ! {p}")
    typer.echo("\nконтекст: " + ", ".join(c.clause_id for c in resp.contexts))
    print("[dim]" + ", ".join(f"{k} {v:.0f} мс" for k, v in resp.timings_ms.items()) + "[/dim]")


if __name__ == "__main__":
    app()
