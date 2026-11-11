from dataclasses import dataclass

from kznorms.eval.eval_retrieval import evaluate, hit_at, recall_at, rr_at
from kznorms.schemas import Clause, EvalItem


def test_hit_recall_rr():
    ranked = ["a", "b", "c", "d", "e", "f"]
    assert hit_at(ranked, ["c"], 5) == 1.0
    assert hit_at(ranked, ["f"], 5) == 0.0
    assert recall_at(ranked, ["b", "f"], 5) == 0.5  # один из двух gold в top-5
    assert rr_at(ranked, ["c"]) == 1 / 3
    assert rr_at(ranked, ["z"]) == 0.0
    assert rr_at(ranked, ["f", "b"]) == 1 / 2  # считается первый найденный gold


@dataclass
class FakeHit:
    clause: Clause
    score: float
    chunk: str = ""


class FakeRetriever:
    """Всегда возвращает a, b, c — проверяем арифметику evaluate без моделей."""

    timings_ms = {"bm25": 1.0}

    def retrieve(self, query, k=10, mode=None):
        mk = lambda cid: Clause(clause_id=cid, doc_code="", doc_title="", number="", text="")  # noqa: E731
        return [FakeHit(mk(c), s) for c, s in (("a", 0.9), ("b", 0.5), ("c", 0.1))]


def test_evaluate_skips_unanswerable_in_metrics():
    items = [
        EvalItem(id="1", question="?", answerable=True, gold_clause_ids=["a"]),
        EvalItem(id="2", question="?", answerable=True, gold_clause_ids=["z"]),
        EvalItem(id="3", question="?", answerable=False),
    ]
    res = evaluate(FakeRetriever(), items, mode="bm25")
    assert res.hit5 == 0.5 and res.mrr10 == 0.5
    assert [m["id"] for m in res.misses] == ["2"]
    assert res.top1_unanswerable == [0.9]
