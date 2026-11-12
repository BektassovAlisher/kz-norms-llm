from kznorms.eval.eval_generation import GenRecord, failures, oracle_hits, summarize
from kznorms.retrieval.chunking import Chunk
from kznorms.retrieval.retriever import Hit
from kznorms.schemas import Clause


def clause(cid: str) -> Clause:
    return Clause(clause_id=cid, doc_code="СН", doc_title="Т", number=cid[2:], text=f"текст {cid}")


class FakeReranker:
    def score(self, query, texts):
        return [float("нужный" in t) for t in texts]


class FakeRetriever:
    def __init__(self) -> None:
        ids = ["d:1", "d:2", "d:3", "d:4", "d:5", "d:6", "d:9"]
        self.clauses = {i: clause(i) for i in ids}
        self.clause_chunks = {i: [Chunk(f"{i}#0", i, f"Т\nтекст {i}")] for i in ids}
        self.clause_chunks["d:9"] = [
            Chunk("d:9#0", "d:9", "Т\nначало таблицы"),
            Chunk("d:9#1", "d:9", "Т\nнужный фрагмент"),
        ]
        self.reranker = FakeReranker()

    def retrieve(self, query, k=None):
        return [Hit(self.clauses[f"d:{i}"], 1 / i, f"Т\nтекст d:{i}") for i in range(1, k + 1)]


def test_oracle_context_has_gold_and_k_passages():
    r = FakeRetriever()
    hits = oracle_hits(r, "вопрос", ["d:9"], k=5)
    ids = [h.clause.clause_id for h in hits]
    assert len(ids) == 5 and "d:9" in ids and len(set(ids)) == 5
    # gold не нашёлся поиском — берём его лучший фрагмент по реранкеру
    assert next(h for h in hits if h.clause.clause_id == "d:9").chunk.endswith("нужный фрагмент")
    # порядок воспроизводим
    assert ids == [h.clause.clause_id for h in oracle_hits(r, "вопрос", ["d:9"], k=5)]


def test_oracle_gold_found_by_search_is_not_duplicated():
    hits = oracle_hits(FakeRetriever(), "вопрос", ["d:2"], k=5)
    assert sorted(h.clause.clause_id for h in hits) == ["d:1", "d:2", "d:3", "d:4", "d:5"]


def rec(answerable, status, cited=True, problems=(), gold_answer="Не менее 1,05 м.", answer=""):
    cit = [{"clause_id": "d:1" if cited else "d:2", "quote": "x"}]
    return GenRecord(
        id="q", mode="oracle", answerable=answerable, type="", question="?",
        gold_answer=gold_answer if answerable else "",
        gold_clause_ids=["d:1"] if answerable else [],
        context_ids=["d:1", "d:2"],
        status=status, answer=answer,
        citations=cit if status != "not_found" else [],
        problems=list(problems), llm_ms=100.0,
    )  # fmt: skip


def test_summary_metrics():
    records = [
        rec(True, "answered", answer="Не менее 1,05 м."),
        rec(True, "partial", cited=False, answer="1,2 м", problems=["число"]),
        rec(True, "not_found"),  # ложный отказ
        rec(False, "not_found"),
        rec(False, "answered", answer="Да."),  # ответила на вопрос без ответа
    ]
    m = summarize(records, "oracle").metrics
    assert m["answered"] == 2 / 3 and m["answered_if_gold"] == 2 / 3
    assert m["refusal_ok"] == 0.5
    assert m["verify_ok"] == 2 / 3  # среди данных ответов: 3 шт., один с проблемой
    assert m["gold_cited"] == 0.5
    assert m["numbers"] == 1 / 3  # 1.05 есть / нет / отказ
    assert m["status_answered"] == 0.5
    f = failures(records)
    assert len(f["ложный отказ (gold был в контексте)"]) == 1
    assert len(f["ответ на вопрос без ответа"]) == 1


def test_reverify_roundtrip(tmp_path):
    from kznorms.eval.eval_generation import load_records, reverify, save_records

    r = rec(True, "answered", answer="Не менее 1,05 м.", problems=["старая проблема"])
    r.citations = [{"clause_id": "d:1", "quote": "не менее 1,05 м"}]
    r.passages = [{"clause_id": "d:1", "doc_code": "СН", "number": "1", "text": "Не менее 1,05 м."}]
    save_records([r], tmp_path / "g.jsonl")
    loaded = load_records(tmp_path / "g.jsonl")
    reverify(loaded)
    assert loaded[0].problems == [] and loaded[0].cited_gold
