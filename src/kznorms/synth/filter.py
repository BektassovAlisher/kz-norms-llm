"""Фильтр через verify.py, дедуп вопросов, отсев длинных ответов.

Учитель тоже ошибается, а ученик выучит его ошибки как правило. Поэтому в обучение идёт
только то, что проверяемо правильно по форме:
- пример с gold: статус не not_found, verify пройден, процитирован именно gold, ответ не
  длиннее лимита (краткость — один из навыков, которым учим);
- пример без gold: учитель сказал not_found. Если ответил — значит, в контексте нашёлся
  другой пункт с ответом (дубль, который не поймал jaccard), и «отказ» был бы ложью.
"""

from kznorms.rag.prompts import Passage
from kznorms.rag.verify import verify
from kznorms.schemas import SynthExample
from kznorms.synth.build_contexts import jaccard


def check(ex: SynthExample, max_answer_chars: int) -> list[str]:
    a = ex.answer
    if a is None:
        return ["нет ответа по схеме"]
    if ex.not_found:
        return [] if a.status == "not_found" else ["учитель нашёл ответ без gold"]
    if a.status == "not_found":
        return ["учитель отказался при gold в контексте"]
    passages = [Passage(**p) for p in ex.passages]
    reasons = [f"verify: {p}" for p in verify(a, passages, ex.q.question).problems]
    if not any(c.clause_id == ex.q.clause_id for c in a.citations):
        reasons.append("gold не процитирован")
    if len(a.answer) > max_answer_chars:
        reasons.append("слишком длинный ответ")
    return reasons


def dedup(examples: list[SynthExample], threshold: float) -> None:
    """Почти одинаковые вопросы к одному пункту — помечаем дублями (оставляем первый)."""
    kept: dict[str, list[str]] = {}
    for ex in examples:
        if ex.reasons:
            continue
        same = kept.setdefault(ex.q.clause_id, [])
        if any(jaccard(ex.q.question, other) >= threshold for other in same):
            ex.reasons.append("дубль вопроса")
        else:
            same.append(ex.q.question)


def apply_filter(examples: list[SynthExample], cfg: dict) -> list[SynthExample]:
    for ex in examples:
        ex.reasons = check(ex, cfg["max_answer_chars"])
    dedup(examples, cfg["dedup_jaccard"])
    return [ex for ex in examples if not ex.reasons]
