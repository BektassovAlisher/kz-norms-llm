"""Оценка генерации: oracle и end-to-end.

Два режима отвечают на разные вопросы:
- oracle — в контексте гарантированно есть gold-пункты (+ «трудные» дистракторы из поиска,
  в перемешанном порядке). Ошибки поиска исключены — меряем только LLM. Именно этот режим
  показывает эффект дообучения в чистом виде.
- e2e — контекст как в проде, из поиска. Меряем систему целиком; ошибка может быть и
  поиска, и модели — поэтому отдельно считаем «ответила, когда gold был в контексте».

Для вопросов без ответа gold нет, и контекст в обоих режимах один — top-k поиска.

Метрики автоматические, без LLM-судьи (он появится вместе с API учителя):
- answered      — вопрос с ответом, модель не отказалась (1 − доля ложных отказов);
- refusal_ok    — вопрос без ответа, модель честно сказала not_found;
- verify_ok     — среди данных ответов: прошли verify (ничего не выдумано);
- gold_cited    — среди данных ответов на вопросы с ответом: процитирован хотя бы один gold;
- numbers       — доля чисел эталонного ответа, которые есть в ответе модели (отказ = 0):
                  дешёвый заменитель «правильно ли», работает для вопросов с числами;
- status=answered — среди данных ответов при gold в контексте: насколько модель уверена
                  (базовая модель ставит partial даже идеальным ответам).
"""

import json
import random
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from kznorms.rag.pipeline import RagPipeline, to_passage
from kznorms.rag.prompts import Passage
from kznorms.rag.verify import numbers, verify
from kznorms.retrieval.retriever import Hit
from kznorms.schemas import Answer, EvalItem

MODES = ("oracle", "e2e")


def gold_hit(retriever, question: str, clause_id: str) -> Hit:
    """Gold-пункт как Hit: у длинного пункта берём фрагмент, который реранкер счёл лучшим."""
    chunks = retriever.clause_chunks[clause_id]
    if len(chunks) == 1:
        return Hit(retriever.clauses[clause_id], 1.0, chunks[0].text)
    scores = retriever.reranker.score(question, [c.text for c in chunks])
    best = max(range(len(chunks)), key=lambda i: scores[i])
    return Hit(retriever.clauses[clause_id], float(scores[best]), chunks[best].text)


def oracle_hits(retriever, question: str, gold_ids: list[str], k: int) -> list[Hit]:
    """Gold + трудные дистракторы (top поиска без gold), порядок перемешан.

    Перемешиваем, потому что у моделей есть позиционный перекос: если gold всегда первый,
    модель выучит «отвечай по первому пункту», а не «найди нужный пункт».
    Зерно — сам вопрос: порядок воспроизводим между запусками и моделями.
    """
    found = retriever.retrieve(question, k=k + len(gold_ids))
    by_id = {h.clause.clause_id: h for h in found}
    gold = [by_id.get(g) or gold_hit(retriever, question, g) for g in gold_ids]
    distractors = [h for h in found if h.clause.clause_id not in gold_ids]
    hits = gold + distractors[: max(k - len(gold), 0)]
    random.Random(question).shuffle(hits)
    return hits


@dataclass
class GenRecord:
    id: str
    mode: str
    answerable: bool
    type: str
    question: str
    gold_answer: str
    gold_clause_ids: list[str]
    context_ids: list[str]
    status: str
    answer: str
    citations: list[dict]
    problems: list[str]
    llm_ms: float
    passages: list[dict] = field(default_factory=list)  # контекст — для --reverify
    gold_in_context: bool = False
    cited_gold: bool = False
    numbers_recall: float | None = None  # None — в эталоне нет чисел

    def __post_init__(self) -> None:
        self.gold_in_context = any(g in self.context_ids for g in self.gold_clause_ids)
        self.cited_gold = any(c["clause_id"] in self.gold_clause_ids for c in self.citations)
        gold_nums = numbers(self.gold_answer)
        if self.answerable and gold_nums:
            got = numbers(self.answer) if self.status != "not_found" else set()
            self.numbers_recall = len(gold_nums & got) / len(gold_nums)


def run_item(pipe: RagPipeline, it: EvalItem, mode: str, k: int) -> GenRecord:
    if mode == "oracle" and it.answerable:
        hits = oracle_hits(pipe.retriever, it.question, it.gold_clause_ids, k)
    else:
        hits = pipe.retriever.retrieve(it.question, k=k)
    passages = [to_passage(h) for h in hits]
    gen = pipe.generate(it.question, passages)
    return GenRecord(
        id=it.id,
        mode=mode,
        answerable=it.answerable,
        type=it.type,
        question=it.question,
        gold_answer=it.gold_answer,
        gold_clause_ids=it.gold_clause_ids,
        context_ids=[h.clause.clause_id for h in hits],
        status=gen.answer.status,
        answer=gen.answer.answer,
        citations=[c.model_dump() for c in gen.answer.citations],
        problems=gen.problems,
        llm_ms=gen.llm_ms,
        passages=[asdict(p) for p in passages],
    )


def _share(xs: list[bool]) -> float | None:
    return sum(xs) / len(xs) if xs else None


@dataclass
class GenSummary:
    mode: str
    n_ans: int = 0
    n_unans: int = 0
    metrics: dict[str, float | None] = field(default_factory=dict)
    statuses: dict[str, int] = field(default_factory=dict)


def summarize(records: list[GenRecord], mode: str) -> GenSummary:
    rs = [r for r in records if r.mode == mode]
    ans = [r for r in rs if r.answerable]
    unans = [r for r in rs if not r.answerable]
    given = [r for r in rs if r.status != "not_found"]
    given_ans = [r for r in ans if r.status != "not_found"]
    with_gold = [r for r in ans if r.gold_in_context]
    nums = [r.numbers_recall for r in ans if r.numbers_recall is not None]
    m = {
        "answered": _share([r.status != "not_found" for r in ans]),
        "answered_if_gold": _share([r.status != "not_found" for r in with_gold]),
        "refusal_ok": _share([r.status == "not_found" for r in unans]),
        "verify_ok": _share([not r.problems for r in given]),
        "gold_cited": _share([r.cited_gold for r in given_ans]),
        "numbers": statistics.mean(nums) if nums else None,
        "status_answered": _share(
            [r.status == "answered" for r in with_gold if r.status != "not_found"]
        ),
        "gold_in_context": _share([r.gold_in_context for r in ans]),
        "answer_len_median": statistics.median(len(r.answer) for r in given) if given else None,
        "llm_ms_p50": statistics.median(r.llm_ms for r in rs) if rs else None,
    }
    return GenSummary(mode, len(ans), len(unans), m, dict(Counter(r.status for r in rs)))


LABELS = {
    "answered": "ответила (вопросы с ответом)",
    "answered_if_gold": "ответила, когда gold в контексте",
    "refusal_ok": "честный not_found (вопросы без ответа)",
    "verify_ok": "прошли verify (среди ответов)",
    "gold_cited": "процитирован gold",
    "numbers": "числа эталона в ответе",
    "status_answered": "status=answered при gold в контексте",
    "gold_in_context": "gold в контексте",
    "answer_len_median": "длина ответа, медиана, симв.",
    "llm_ms_p50": "LLM p50, мс",
}


def _fmt(v: float | None, key: str) -> str:
    if v is None:
        return "—"
    return f"{v:.0f}" if key in ("answer_len_median", "llm_ms_p50") else f"{v:.2f}"


def failures(records: list[GenRecord]) -> dict[str, list[GenRecord]]:
    return {
        "ложный отказ (gold был в контексте)": [
            r for r in records if r.answerable and r.gold_in_context and r.status == "not_found"
        ],
        "ответ на вопрос без ответа": [
            r for r in records if not r.answerable and r.status != "not_found"
        ],
        "не прошёл verify": [r for r in records if r.problems],
        "gold не процитирован": [
            r for r in records if r.answerable and r.status != "not_found" and not r.cited_gold
        ],
        "числа эталона не все": [
            r for r in records if r.numbers_recall is not None and r.numbers_recall < 1
        ],
    }


def to_markdown(summaries: list[GenSummary], records: list[GenRecord], model: str, note: str):
    head = summaries[0]
    lines = [
        f"# Генерация: {model}",
        "",
        f"Вопросов с ответом: {head.n_ans}, без ответа: {head.n_unans}. {note}",
        "",
        "| метрика | " + " | ".join(s.mode for s in summaries) + " |",
        "|---|" + "---|" * len(summaries),
    ]
    for key, label in LABELS.items():
        vals = " | ".join(_fmt(s.metrics.get(key), key) for s in summaries)
        lines.append(f"| {label} | {vals} |")
    lines += ["", "status: " + "; ".join(f"{s.mode} {s.statuses}" for s in summaries)]
    for s in summaries:
        for title, rs in failures([r for r in records if r.mode == s.mode]).items():
            if not rs:
                continue
            lines += ["", f"## {s.mode}: {title} — {len(rs)}", ""]
            for r in rs:
                extra = f" — {'; '.join(r.problems)}" if title == "не прошёл verify" else ""
                lines.append(f"- `{r.id}` {r.question} → {r.status}: {r.answer[:160]}{extra}")
    return "\n".join(lines) + "\n"


def reverify(records: list[GenRecord]) -> None:
    """Пересчитать verify по сохранённым ответам и контексту — без LLM.

    Генерация дорогая (минуты на GPU), проверка — миллисекунды. Когда меняется verify,
    перегенерировать ответы не нужно: ответы те же, меняется только их оценка.
    """
    for r in records:
        if r.problems and r.problems[0].startswith(("ответ обрезан", "ответ не по JSON")):
            continue  # ответа нет — проверять нечего
        ans = Answer(status=r.status, answer=r.answer, citations=r.citations)
        r.problems = verify(ans, [Passage(**d) for d in r.passages], r.question).problems


def load_records(path: Path) -> list[GenRecord]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [GenRecord(**json.loads(x)) for x in lines if x.strip()]


def save_records(records: list[GenRecord], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
