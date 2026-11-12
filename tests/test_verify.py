from kznorms.rag.prompts import Passage, build_messages
from kznorms.rag.verify import numbers, verify
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


P2 = [
    *P,
    Passage(
        "doc:5.3.10", "СН РК 3.02-01-2023", "5.3.10", "Пределы огнестойкости I -не менее 1,5 ч."
    ),
]


def test_punctuation_and_spacing_do_not_matter():
    a = ans("answered", "Не менее 1,5 ч.", ("doc:5.3.10", "огнестойкости I - не менее 1,5 ч"))
    assert verify(a, P2).ok


def test_corrected_word_is_not_a_quote():
    a = ans("answered", "Не менее 1,05 м.", ("doc:5.3.9", "ширина марша лестниц должна быть"))
    assert not verify(a, P2).ok


def test_quote_from_another_clause_is_named():
    a = ans("answered", "Не менее 1,5 ч.", ("doc:5.3.9", "не менее 1,5 ч"))
    assert "она из doc:5.3.10" in verify(a, P2).problems[0]


def test_clause_number_of_uncited_passage_is_allowed():
    a = ans("answered", "Не менее 1,05 м; см. также п. 5.3.10.", ("doc:5.3.9", "не менее 1,05 м"))
    assert verify(a, P2).ok


def test_list_markers_are_not_numbers():
    assert numbers("1) не менее 1,05 м; 12) далее; 2) 6 м.") == {"1.05", "6"}


def test_number_in_context_but_not_quoted_vs_invented():
    a = ans("answered", "1,05 м и 1,5 ч, а также 7 м.", ("doc:5.3.9", "не менее 1,05 м"))
    problems = verify(a, P2).problems
    assert problems == [
        "числа из контекста без цитаты: ['1.5']",
        "числа, которых нет в контексте: ['7']",
    ]


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
