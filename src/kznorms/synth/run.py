"""Синтетика целиком: пункты → вопросы → контексты → ответы → фильтр → data/sft/.

Каждый этап пишет свой файл в data/synthetic/ — по ним удобно разбирать, где теряются
примеры. SFT-пример — это ровно промпт прода (`build_messages`) + ответ учителя как текст
ассистента: формат обучения и инференса совпадает символ в символ.
"""

import json
import random
import statistics
from collections import Counter
from pathlib import Path

from kznorms.paths import SFT, SYNTHETIC
from kznorms.rag.prompts import Passage, build_messages
from kznorms.schemas import SynthExample, SynthQuestion
from kznorms.synth.build_contexts import build_contexts
from kznorms.synth.filter import apply_filter
from kznorms.synth.gen_answers import gen_answers
from kznorms.synth.gen_questions import body, gen_questions, lexical_overlap, select_chunks


def to_sft(ex: SynthExample) -> dict:
    passages = [Passage(**p) for p in ex.passages]
    messages = build_messages(ex.q.question, passages)
    messages.append({"role": "assistant", "content": ex.answer.model_dump_json()})
    meta = {"qid": ex.q.qid, "qtype": ex.q.qtype, "not_found": ex.not_found}
    return {"messages": messages, "meta": meta}


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write((r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)) + "\n")


def report(questions, examples, kept, retriever, teacher) -> dict:
    overlaps = [
        lexical_overlap(q.question, body(retriever.chunks[q.chunk_id].text)) for q in questions
    ]
    reasons = Counter(r.split(":")[0] if r.startswith("verify") else r
                      for ex in examples for r in ex.reasons)  # fmt: skip
    return {
        "teacher": teacher.model,
        "questions": len(questions),
        "question_types": dict(Counter(q.qtype for q in questions)),
        "lexical_overlap_median": round(statistics.median(overlaps), 2) if overlaps else None,
        "examples": len(examples),
        "kept": len(kept),
        "kept_by_split": dict(Counter(ex.q.split for ex in kept)),
        "kept_not_found_share": round(sum(ex.not_found for ex in kept) / max(len(kept), 1), 2),
        "kept_status": dict(Counter(ex.answer.status for ex in kept)),
        "kept_relabeled": sum(ex.relabeled for ex in kept),
        "drop_reasons": dict(reasons.most_common()),
        "teacher_calls": {"new": teacher.calls, "cache": teacher.cached, "failed": teacher.failed},
    }


STAGES = ("questions", "contexts", "answers")
QUESTIONS = SYNTHETIC / "questions.jsonl"
CONTEXTS = SYNTHETIC / "contexts.jsonl"


def stage_questions(teacher, retriever, split_of: dict[str, str], cfg: dict) -> None:
    rng = random.Random(cfg["seed"])
    picked = select_chunks(
        retriever, split_of, cfg["n_clauses"], cfg["val_share"],
        cfg["min_passage_chars"], cfg["min_letter_share"], rng,
    )  # fmt: skip
    questions = gen_questions(teacher, retriever, picked, cfg)
    _write_jsonl(QUESTIONS, (q.model_dump_json() for q in questions))


def stage_contexts(retriever, split_of: dict[str, str], cfg: dict) -> None:
    """Только поиск (GPU: BGE-M3 + реранкер), учитель не нужен."""
    allowed = {slug for slug, sp in split_of.items() if sp in ("train", "val")}
    retriever.warmup("hybrid_rerank")
    cache = {ex.q.qid: ex for ex in load_examples(CONTEXTS)} if CONTEXTS.exists() else {}
    examples = build_contexts(retriever, load_questions(), allowed, cfg, cache)
    _write_jsonl(CONTEXTS, (ex.model_dump_json() for ex in examples))


def stage_answers(teacher, retriever, cfg: dict) -> dict:
    """Только учитель; из поиска нужны лишь тексты фрагментов (для отчёта), без GPU."""
    examples = gen_answers(teacher, load_examples(CONTEXTS), cfg)
    kept = apply_filter(examples, cfg)
    _write_jsonl(SYNTHETIC / "examples.jsonl", (ex.model_dump_json() for ex in examples))
    for split in ("train", "val"):
        _write_jsonl(SFT / f"{split}.jsonl", (to_sft(ex) for ex in kept if ex.q.split == split))
    rep = report(load_questions(), examples, kept, retriever, teacher)
    (SYNTHETIC / "report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return rep


def run(teacher, retriever, split_of: dict[str, str], cfg: dict, stages=STAGES) -> dict | None:
    """Этапы по порядку; каждый читает файл предыдущего, так что их можно запускать отдельно.

    Зачем отдельно: локальный учитель (llama-server) и модели поиска вместе не влезают в 8 ГБ.
    Тогда: вопросы (сервер с учителем) → контексты (сервер выключен, GPU у поиска) → ответы.
    """
    if "questions" in stages:
        stage_questions(teacher, retriever, split_of, cfg)
    if "contexts" in stages:
        stage_contexts(retriever, split_of, cfg)
    if "answers" in stages:
        return stage_answers(teacher, retriever, cfg)
    return None


def load_examples(path: Path) -> list[SynthExample]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [SynthExample.model_validate_json(x) for x in lines if x.strip()]


def load_questions(path: Path | None = None) -> list[SynthQuestion]:
    lines = (path or QUESTIONS).read_text(encoding="utf-8").splitlines()
    return [SynthQuestion.model_validate_json(x) for x in lines if x.strip()]
