"""HTTP API на подставном пайплайне — без GPU и llama-server."""

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import APIConnectionError

from kznorms.api.main import create_app
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
GOOD = (
    '{"status": "answered", "answer": "Не менее 1,05 м.", '
    '"citations": [{"clause_id": "doc:5.3.9", "quote": "не менее 1,05 м"}]}'
)


class FakeRetriever:
    def __init__(self) -> None:
        self.timings_ms: dict[str, float] = {}
        self.calls: list[tuple] = []

    def retrieve(self, query, k=None, mode=None):
        self.calls.append((query, k, mode))
        self.timings_ms = {"bm25": 1.0}
        return [Hit(CLAUSE, 0.9, "Здания жилые. 5 ТРЕБОВАНИЯ\n" + CLAUSE.text)]


class FakeLLM:
    model = "fake"
    last_ms = 5.0

    def __init__(self, raw: str = GOOD, error: Exception | None = None) -> None:
        self.raw, self.error = raw, error

    def chat_json(self, messages, schema, max_tokens=1024):
        if self.error:
            raise self.error
        return self.raw


def client(llm: FakeLLM | None = None) -> tuple[TestClient, FakeRetriever]:
    retriever = FakeRetriever()
    pipe = RagPipeline(llm or FakeLLM(), retriever)
    return TestClient(create_app(lambda: pipe)), retriever


def test_health_reports_model():
    c, _ = client()
    with c:
        assert c.get("/health").json() == {"status": "ok", "model": "fake"}


def test_ask_returns_verified_answer_with_citations():
    c, _ = client()
    with c:
        body = c.post("/ask", json={"question": "Ширина марша?"}).json()
    assert body["verified"] is True
    assert body["answer"]["citations"][0]["clause_id"] == "doc:5.3.9"
    assert body["timings_ms"] == {"bm25": 1.0, "llm": 5.0}


def test_ask_flags_invented_quote():
    raw = GOOD.replace("1,05", "1,2")
    c, _ = client(FakeLLM(raw))
    with c:
        body = c.post("/ask", json={"question": "Ширина марша?"}).json()
    assert body["verified"] is False and body["problems"]


def test_search_passes_mode_and_k():
    c, retriever = client()
    with c:
        body = c.post("/search", json={"question": "Ширина марша?", "k": 3, "mode": "bm25"}).json()
    assert retriever.calls == [("Ширина марша?", 3, "bm25")]
    assert body["hits"][0]["clause"]["clause_id"] == "doc:5.3.9"


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "a"},
        {"question": "Ширина марша?", "k": 100},
        {"question": "Ширина марша?", "mode": "rag"},
    ],
)
def test_bad_input_is_rejected(payload):
    c, _ = client()
    with c:
        assert c.post("/search", json=payload).status_code == 422


def test_llm_down_gives_503():
    error = APIConnectionError(request=httpx.Request("POST", "http://127.0.0.1:8085"))
    c, _ = client(FakeLLM(error=error))
    with c:
        r = c.post("/ask", json={"question": "Ширина марша?"})
    assert r.status_code == 503 and "llama-server" in r.json()["detail"]
