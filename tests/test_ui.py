from kznorms.api.ui import highlight, render_answer, render_contexts
from kznorms.schemas import Answer, AnswerResponse, Citation, Clause

CL = Clause(
    clause_id="doc:5.3.9",
    doc_code="СН РК 3.02-01-2023",
    doc_title="Здания жилые",
    section="5 ТРЕБОВАНИЯ",
    number="5.3.9",
    text="Ширина марша лестницы должна быть не менее 1,05 м. Уклон <1:2>.",
)


def resp(status="answered", quote="не менее 1,05 м", problems=()):
    cits = [Citation(clause_id="doc:5.3.9", quote=quote)] if status != "not_found" else []
    return AnswerResponse(
        answer=Answer(status=status, answer="Не менее 1,05 м.", citations=cits),
        contexts=[CL],
        verified=not problems,
        problems=list(problems),
    )


def test_highlight_escapes_and_marks_case_insensitive():
    out = highlight(CL.text, ["НЕ МЕНЕЕ 1,05 м", "отсутствует"])
    assert "<mark>не менее 1,05 м</mark>" in out
    assert "&lt;1:2&gt;" in out and "<1:2>" not in out  # текст нормы экранирован


def test_highlight_overlapping_quotes():
    out = highlight("abcdef", ["abcd", "cdef"])
    assert out == "<mark>abcd</mark><mark>ef</mark>"


def test_render_answer_badges_and_citation():
    out = render_answer(resp())
    assert "ответ найден" in out and "verify ✓" in out
    assert "СН РК 3.02-01-2023, п. 5.3.9" in out
    bad = render_answer(resp(problems=["цитата не найдена в doc:5.3.9: 'x'"]))
    assert "verify ✗" in bad and "Проверка не пройдена" in bad and "&#x27;x&#x27;" in bad


def test_render_not_found_has_no_verify_badge():
    out = render_answer(resp(status="not_found"))
    assert "в нормах не найдено" in out and "verify" not in out


def test_cited_context_is_open_and_marked():
    out = render_contexts(resp())
    assert '<details class="kz-ctx" open>' in out and "★" in out and "<mark>" in out
