"""Recall@5, MRR@10 по режимам поиска.

Метрики считаются только по вопросам, у которых ответ есть (answerable): для вопроса без
ответа «правильного пункта» не существует. Зато по ним видно другое — насколько уверен
реранкер, когда отвечать нечем. Если top-1 оценка у вопросов без ответа заметно ниже, порог
по ней — дешёвый первый фильтр для not_found ещё до LLM.

- Hit@k     — доля вопросов, где хотя бы один gold-пункт попал в top-k.
- Recall@k  — доля gold-пунктов, попавших в top-k (важно для вопросов на несколько пунктов).
- MRR@10    — среднее 1/место первого gold-пункта (0, если не попал в top-10):
              Hit говорит «нашли или нет», MRR — «насколько высоко».
"""

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from kznorms.paths import EVAL
from kznorms.schemas import EvalItem

QA_TEST = EVAL / "qa_test.jsonl"


def load_items(path: Path = QA_TEST) -> list[EvalItem]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [EvalItem.model_validate_json(x) for x in lines if x.strip()]


def save_items(items: list[EvalItem], path: Path = QA_TEST) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(it.model_dump_json() + "\n")


def hit_at(ranked: list[str], gold: list[str], k: int) -> float:
    return float(any(g in ranked[:k] for g in gold))


def recall_at(ranked: list[str], gold: list[str], k: int) -> float:
    return len(set(gold) & set(ranked[:k])) / len(gold)


def rr_at(ranked: list[str], gold: list[str], k: int = 10) -> float:
    for rank, cid in enumerate(ranked[:k], start=1):
        if cid in gold:
            return 1.0 / rank
    return 0.0


@dataclass
class ModeResult:
    mode: str
    hit5: float = 0.0
    recall5: float = 0.0
    mrr10: float = 0.0
    latency_ms_p50: float = 0.0
    top1_answerable: list[float] = field(default_factory=list)
    top1_unanswerable: list[float] = field(default_factory=list)
    misses: list[dict] = field(default_factory=list)  # для разбора ошибок глазами


def evaluate(retriever, items: list[EvalItem], mode: str, k: int = 10) -> ModeResult:
    res = ModeResult(mode)
    hits, recalls, rrs, lat = [], [], [], []
    for it in items:
        found = retriever.retrieve(it.question, k=k, mode=mode)
        lat.append(sum(retriever.timings_ms.values()))
        ranked = [h.clause.clause_id for h in found]
        top1 = found[0].score if found else float("nan")
        if not it.answerable:
            res.top1_unanswerable.append(top1)
            continue
        res.top1_answerable.append(top1)
        hits.append(hit_at(ranked, it.gold_clause_ids, 5))
        recalls.append(recall_at(ranked, it.gold_clause_ids, 5))
        rrs.append(rr_at(ranked, it.gold_clause_ids, 10))
        if not hits[-1]:
            res.misses.append(
                {
                    "id": it.id,
                    "question": it.question,
                    "gold": it.gold_clause_ids,
                    "top5": ranked[:5],
                }
            )
    if hits:
        res.hit5, res.recall5, res.mrr10 = map(statistics.mean, (hits, recalls, rrs))
    res.latency_ms_p50 = statistics.median(lat) if lat else 0.0
    return res


def to_markdown(results: list[ModeResult], n_ans: int, n_unans: int, note: str) -> str:
    lines = [
        "# Поиск: метрики по режимам",
        "",
        f"Вопросов с ответом: {n_ans}, без ответа: {n_unans}. {note}",
        "",
        "| режим | Hit@5 | Recall@5 | MRR@10 | задержка p50, мс |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.mode} | {r.hit5:.3f} | {r.recall5:.3f} | {r.mrr10:.3f} "
            f"| {r.latency_ms_p50:.0f} |"
        )
    rr = next((r for r in results if r.mode == "hybrid_rerank"), None)
    if rr and rr.top1_answerable and rr.top1_unanswerable:
        a, u = rr.top1_answerable, rr.top1_unanswerable
        lines += [
            "",
            "## Оценка реранкера у top-1",
            "",
            "| вопросы | медиана | мин | макс |",
            "|---|---|---|---|",
            f"| с ответом | {statistics.median(a):.3f} | {min(a):.3f} | {max(a):.3f} |",
            f"| без ответа | {statistics.median(u):.3f} | {min(u):.3f} | {max(u):.3f} |",
        ]
    for r in results:
        if r.misses:
            lines += ["", f"## Промахи {r.mode} (gold нет в top-5)", ""]
            for m in r.misses:
                lines.append(
                    f"- `{m['id']}` {m['question']} — gold {m['gold']}, "
                    f"top-5: {', '.join(m['top5'])}"
                )
    return "\n".join(lines) + "\n"


def write_misses(results: list[ModeResult], path: Path) -> None:
    path.write_text(
        json.dumps({r.mode: r.misses for r in results}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
