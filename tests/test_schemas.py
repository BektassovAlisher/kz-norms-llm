import pytest
from pydantic import ValidationError

from kznorms.schemas import Answer, Clause


def test_answer_roundtrip():
    raw = '{"status":"answered","answer":"…","citations":[{"clause_id":"doc1:4.2.5","quote":"…"}]}'
    ans = Answer.model_validate_json(raw)
    assert ans.citations[0].clause_id == "doc1:4.2.5"


def test_answer_rejects_bad_status():
    with pytest.raises(ValidationError):
        Answer(status="maybe", answer="", citations=[])


def test_clause_minimal():
    c = Clause(clause_id="doc1:4.2.5", doc_code="СП РК …", doc_title="…", number="4.2.5", text="…")
    assert c.status == "действует"
