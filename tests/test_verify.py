from kznorms.rag.prompts import Passage, build_messages
from kznorms.rag.verify import verify
from kznorms.schemas import Answer, Citation

P = [
    Passage(
        "doc:5.3.9",
        "СН РК 3.02-01-2023",
        "5.3.9",
        "Ширина марша лестницы должна быть не менее 1,05 м.",
    )
]


def ans(status: str, text: str, *cits: tuple[str, str]) -> Answer:
    citations = [Citation(clause_id=i, quote=q) for i, q in cits]
    return Answer(status=status, answer=text, citations=citations)


def test_good_answer_passes():
    a = ans("answered", "Не менее 1,05 м (п. 5.3.9).", ("doc:5.3.9", "не менее 1,05 м"))
    assert verify(a, P).ok


def test_quote_with_other_whitespace_passes():
    a = ans("answered", "Ширина задаётся нормой.", ("doc:5.3.9", "марша  лестницы\nдолжна"))
    assert verify(a, P).ok


def test_invented_clause_fails():
    a = ans("answered", "Не менее 1,05 м.", ("doc:5.3.10", "не менее 1,05 м"))
    assert not verify(a, P).ok


def test_paraphrase_instead_of_quote_fails():
    a = ans("answered", "Не менее 1,05 м.", ("doc:5.3.9", "ширина не меньше 1,05 м"))
    assert not verify(a, P).ok


def test_empty_quote_fails():
    assert not verify(ans("answered", "Да.", ("doc:5.3.9", " ")), P).ok


def test_number_from_nowhere_fails():
    a = ans("answered", "Не менее 1,2 м.", ("doc:5.3.9", "не менее 1,05 м"))
    assert "1.2" in verify(a, P).problems[-1]


def test_number_from_question_is_allowed():
    a = ans("answered", "На высоте 20 м — не менее 1,05 м.", ("doc:5.3.9", "не менее 1,05 м"))
    assert not verify(a, P).ok
    assert verify(a, P, question="А если дом на 20 м?").ok


def test_not_found_with_citations_fails():
    assert not verify(ans("not_found", "Нет.", ("doc:5.3.9", "не менее")), P).ok
    assert verify(ans("not_found", "В нормах нет."), P).ok


def test_answered_without_citations_fails():
    assert not verify(ans("answered", "Да, можно."), P).ok


def test_prompt_has_context_and_question():
    msgs = build_messages("Какая ширина марша?", P)
    assert msgs[0]["role"] == "system"
    assert "[doc:5.3.9] СН РК 3.02-01-2023, п. 5.3.9" in msgs[1]["content"]
    assert msgs[1]["content"].endswith("ВОПРОС: Какая ширина марша?")
    assert "(пусто)" in build_messages("?", [])[1]["content"]
