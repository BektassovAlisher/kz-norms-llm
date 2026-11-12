"""Фильтр через verify.py, дедуп вопросов, отсев длинных ответов.

Учитель тоже ошибается, а ученик выучит его ошибки как правило. Поэтому в обучение идёт
только то, что проверяемо правильно по форме:
- пример с gold: статус не not_found, verify пройден, процитирован именно gold, ответ не
  длиннее лимита (краткость — один из навыков, которым учим);
- пример без gold: учитель сказал not_found. Если ответил с цитатой — в контексте нашёлся
  другой пункт с опорой (дубль, который не поймал фильтр контекста), и «отказ» был бы ложью.

Исправление метки. Слабый учитель (Qwen3-8B) часто отказывается «мягко»: status partial
и текст «не указано / не хватает информации» — без единой цитаты. В примере без gold это
и есть отказ: gold мы убрали сами, пункты, которые реранкер считает почти равными gold,
тоже, а опереться учителю не на что. Такой ответ переписываем в not_found с единым текстом.
Ответ без gold, но с цитатой, не трогаем — выбрасываем: там учитель нашёл опору.
"""

from kznorms.rag.prompts import Passage
from kznorms.rag.verify import verify
from kznorms.schemas import Answer, SynthExample
from kznorms.synth.build_contexts import jaccard

NOT_FOUND_TEXT = "В переданных пунктах норм ответа на этот вопрос нет."


def relabel_soft_refusal(ex: SynthExample) -> None:
    """Пример без gold и ответ без цитат → цель not_found (см. docstring модуля)."""
    a = ex.answer
    if ex.not_found and a is not None and not a.citations:
        if a.status != "not_found":
            ex.relabeled = True
        ex.answer = Answer(status="not_found", answer=NOT_FOUND_TEXT)


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
        relabel_soft_refusal(ex)
        ex.reasons = check(ex, cfg["max_answer_chars"])
    dedup(examples, cfg["dedup_jaccard"])
    return [ex for ex in examples if not ex.reasons]
