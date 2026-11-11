from kznorms.retrieval.bm25 import BM25Index, tokenize
from kznorms.retrieval.chunking import make_chunks, split_text
from kznorms.retrieval.hybrid import collapse, rrf
from kznorms.schemas import Clause


def clause(number: str, text: str, section: str = "5 ТРЕБОВАНИЯ") -> Clause:
    return Clause(
        clause_id=f"doc:{number}",
        doc_code="СН РК тест",
        doc_title="Здания жилые",
        section=section,
        number=number,
        text=text,
    )


def test_tokenize_lemmatizes_and_drops_stopwords():
    assert tokenize("Ширина лестниц в зданиях") == ["ширина", "лестница", "здание"]


def test_tokenize_numbers():
    # «не»/«по» — стоп-слова; «менее» оставляем: «не менее» vs «не более» — разный смысл
    assert tokenize("не менее 1,2 м по п. 5.3.2") == ["менее", "1.2", "м", "п", "5.3.2"]
    assert tokenize("Ёлка") == tokenize("елка")


def test_short_text_is_one_chunk():
    assert split_text("Короткий пункт.", max_chars=100) == ["Короткий пункт."]


def test_long_text_split_with_overlap():
    sents = [f"Предложение номер {i} про лестницы." for i in range(40)]
    parts = split_text(" ".join(sents), max_chars=200, overlap=60)
    assert len(parts) > 1
    assert all(len(p) <= 200 for p in parts)
    # последнее предложение куска повторяется в начале следующего
    assert parts[1].startswith(parts[0].split(". ")[-1].rstrip("."))
    # ни одно предложение не потеряно
    joined = " ".join(parts)
    assert all(s in joined for s in sents)


def test_oversized_sentence_is_hard_split():
    parts = split_text("слово " * 100, max_chars=50, overlap=0)
    assert all(len(p) <= 50 for p in parts)


def test_chunks_have_header_and_ids():
    chunks = make_chunks([clause("5.1", "Текст пункта.")])
    assert chunks[0].chunk_id == "doc:5.1#0"
    assert chunks[0].text == "Здания жилые. 5 ТРЕБОВАНИЯ\nТекст пункта."


def test_rrf_prefers_agreement():
    fused = rrf([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert [d for d, _ in fused[:2]] in (["a", "b"], ["b", "a"])
    assert fused[0][1] == 1 / 61 + 1 / 62
    assert {d for d, _ in fused} == {"a", "b", "c", "d"}


def test_collapse_keeps_best_rank_per_clause():
    c2c = {"x#0": "x", "x#1": "x", "y#0": "y"}
    assert collapse([("x#1", 9), ("y#0", 8), ("x#0", 7)], c2c) == ["x", "y"]


def test_bm25_finds_by_word_form_and_roundtrips(tmp_path):
    chunks = make_chunks(
        [
            clause("5.1", "Ширина марша лестницы должна быть не менее 1,05 м."),
            clause("5.2", "Высота помещений принимается не менее 2,7 м."),
            clause("5.3", "Лифты предусматриваются в зданиях высотой более пяти этажей."),
        ]
    )
    idx = BM25Index.build(chunks)
    assert idx.search("какая ширина лестничных маршей")[0][0] == "doc:5.1#0"
    assert idx.search("квантовый компьютер") == []

    idx.save(tmp_path / "bm25")
    loaded = BM25Index.load(tmp_path / "bm25")
    assert loaded.search("высота помещения")[0][0] == "doc:5.2#0"
