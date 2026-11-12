import pytest

from kznorms.ingest.download import load_sources
from kznorms.schemas import EvalItem, Source
from kznorms.splits import check_sources, item_split, load_splits


def src(slug: str, group: str, split: str) -> Source:
    return Source(doc_slug=slug, doc_code=slug, title=slug, format="pdf", group=group, split=split)


def test_real_sources_are_split_correctly():
    sources = load_sources()
    assert check_sources(sources) == []
    assert {s.split for s in sources} == {"train", "val", "test"}


def test_group_torn_between_splits_is_an_error():
    sources = [src("sn-a", "a", "train"), src("sp-a", "a", "test"), src("b", "b", "val")]
    assert check_sources(sources) == ["группа a разорвана между сплитами: ['test', 'train']"]
    with pytest.raises(ValueError):
        load_splits(sources)


def test_missing_split_is_an_error():
    assert check_sources([src("a", "a", "")]) == ["a: не указаны group/split"]


def test_item_split():
    splits = {"sn-a": "test", "sp-a": "test", "b": "train"}

    def item(*gold: str) -> EvalItem:
        return EvalItem(id="q", question="?", answerable=bool(gold), gold_clause_ids=list(gold))

    assert item_split(item("sn-a:1.1", "sp-a:4.2"), splits) == "test"
    assert item_split(item("b:5.1"), splits) == "train"
    assert item_split(item("sn-a:1.1", "b:5.1"), splits) == "mixed"
    assert item_split(item(), splits) == "test"  # без ответа — утечки нет
