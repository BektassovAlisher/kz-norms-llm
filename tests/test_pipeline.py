"""Пайплайн целиком на подставных поиске и LLM — без GPU и llama-server."""

from kznorms.rag.pipeline import RagPipeline
from kznorms.retrieval.retriever import Hit
from kznorms.schemas import Clause

CLAUSE = Clause(
    clause_id="doc:5.3.9",
    doc_code="СН РК 3.02-01-2023",
    doc_title="Здания жилые",
    number="5.3.9",
    text="Ширина марша лестницы должна быть не менее 1,05 м.",
)


class FakeRetriever:
    timings_ms = {"bm25": 1.0}

    def retrieve(self, query, k=None):
        chunk = "Здания жилые. 5 ТРЕБОВАНИЯ\n" + CLAUSE.text
        return [Hit(CLAUSE, 0.9, chunk)]


class FakeLLM:
    model = "fake"
    last_ms = 5.0

    def __init__(self, raw: str) -> None:
        self.raw = raw
        self.messages = None

    def chat_json(self, messages, schema, max_tokens=1024):
        self.messages = messages
        return self.raw


def run(raw: str):
    llm = FakeLLM(raw)
    resp = RagPipeline(llm, FakeRetriever()).answer("Ширина марша?")
    return resp, llm


def test_verified_answer():
    resp, llm = run(
        '{"status": "answered", "answer": "Не менее 1,05 м.", '
        '"citations": [{"clause_id": "doc:5.3.9", "quote": "не менее 1,05 м"}]}'
    )
    assert resp.verified and resp.problems == []
    assert resp.timings_ms == {"bm25": 1.0, "llm": 5.0}
    # заголовок фрагмента в промпт не попадает — там шапка [clause_id] код, п. номер
    assert "5 ТРЕБОВАНИЯ" not in llm.messages[1]["content"]


def test_invented_quote_is_flagged():
    resp, _ = run(
        '{"status": "answered", "answer": "Не менее 1,2 м.", '
        '"citations": [{"clause_id": "doc:5.3.9", "quote": "не менее 1,2 м"}]}'
    )
    assert not resp.verified and resp.answer.answer == "Не менее 1,2 м."


def test_broken_json_becomes_not_found():
    resp, _ = run('{"status": "answered", "answer": ')
    assert resp.answer.status == "not_found" and not resp.verified
    assert "JSON" in resp.problems[0]


def test_get_llm_checks_adapter_on_server(monkeypatch):
    import pytest

    import kznorms.rag.llm_client as lc

    monkeypatch.setattr(lc, "active_adapters", lambda url: ["artifacts/lora/pilot-f16.gguf"])
    assert lc.get_llm("ft").model == "ft:pilot-f16"
    with pytest.raises(SystemExit, match="адаптер"):
        lc.get_llm("base")  # база на сервере с адаптером — ошибка, а не тихая подмена
    monkeypatch.setattr(lc, "active_adapters", lambda url: [])
    assert lc.get_llm("base").model == "local"
    with pytest.raises(SystemExit, match="--lora"):
        lc.get_llm("ft")
