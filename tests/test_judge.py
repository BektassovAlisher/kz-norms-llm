import json

import pytest

from kznorms.eval.eval_generation import GenRecord
from kznorms.eval.judge import (
    check_judge_model,
    judge_messages,
    judge_record,
    parse_verdict,
    summarize_judgments,
)


def rec(status="answered", answer="1,80 м", answerable=True, mode="oracle"):
    return GenRecord(
        id="q1", mode=mode, answerable=answerable, type="", question="Ширина марша в школе?",
        gold_answer="1,80 м для зданий Ф4.1 в 4–5 этажей." if answerable else "",
        gold_clause_ids=["d:1"] if answerable else [], context_ids=["d:1", "d:2"],
        status=status, answer=answer, citations=[], problems=[], llm_ms=1.0,
        passages=[{"clause_id": "d:1", "doc_code": "СП", "number": "1", "text": "ТЕКСТ GOLD"},
                  {"clause_id": "d:2", "doc_code": "СП", "number": "2", "text": "ДРУГОЙ"}],
    )  # fmt: skip


class FakeJudge:
    def __init__(self, raw):
        self.raw, self.calls = raw, 0

    def chat_json(self, messages, schema, temperature=0.0, max_tokens=512):
        self.calls += 1
        return self.raw


def test_prompt_has_gold_answer_and_only_gold_clause():
    user = judge_messages(rec())[1]["content"]
    assert "ЭТАЛОН: 1,80 м для зданий Ф4.1" in user
    assert "ТЕКСТ GOLD" in user and "ДРУГОЙ" not in user
    assert "статус answered" in user


def test_verdicts():
    raw = {"reason": "-", "contradicts": False, "missing": ["Ф4.1", "4–5 этажей"]}
    j = judge_record(FakeJudge(json.dumps(raw, ensure_ascii=False)), rec())
    assert j.verdict == "неполно" and j.reason == "нет: Ф4.1; 4–5 этажей"
    ok = {"reason": "всё есть", "contradicts": False, "missing": [" "]}
    assert judge_record(FakeJudge(json.dumps(ok)), rec()).verdict == "верно"
    bad = {"reason": "-", "contradicts": True, "missing": []}
    assert judge_record(FakeJudge(json.dumps(bad)), rec()).verdict == "неверно"
    refusal = FakeJudge("{}")
    assert judge_record(refusal, rec(status="not_found")).verdict == "отказ"
    assert refusal.calls == 0  # отказ не спрашиваем у судьи
    assert judge_record(FakeJudge("{}"), rec(answerable=False)) is None
    assert parse_verdict("мусор")[0] == "ошибка судьи"
    assert parse_verdict('{"verdict": "отлично"}')[0] == "ошибка судьи"  # нет полей


def test_summary_shares():
    js = [
        judge_record(FakeJudge('{"reason": "", "contradicts": false, "missing": []}'), rec()),
        judge_record(FakeJudge('{"reason": "", "contradicts": true, "missing": []}'), rec()),
        judge_record(FakeJudge(""), rec(status="not_found")),
        judge_record(FakeJudge('{"contradicts": false, "missing": []}'), rec(mode="e2e")),
    ]
    s = summarize_judgments(js, "oracle")
    assert s["верно"] == pytest.approx(1 / 3) and s["отказ"] == pytest.approx(1 / 3)


def test_student_cannot_judge_itself():
    with pytest.raises(SystemExit):
        check_judge_model("Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
    check_judge_model("Qwen3-8B-Q4_K_M.gguf")
