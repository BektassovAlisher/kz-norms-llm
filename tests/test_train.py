"""Подготовка данных для обучения — без GPU и без Unsloth."""

import json

import pytest

from kznorms.train.sft_unsloth import check_mask, load_sft, response_text


def write_rows(path, n):
    path.write_text(
        "\n".join(json.dumps({"messages": [], "meta": {"qid": f"q{i}"}}) for i in range(n)),
        encoding="utf-8",
    )


def test_subsets_are_nested_and_deterministic(tmp_path):
    p = tmp_path / "train.jsonl"
    write_rows(p, 100)
    small = [r["meta"]["qid"] for r in load_sft(p, 10, seed=1)]
    big = [r["meta"]["qid"] for r in load_sft(p, 50, seed=1)]
    assert big[:10] == small  # 500 ⊂ 1000 ⊂ 2000: кривая обучения на одних и тех же данных
    assert small == [r["meta"]["qid"] for r in load_sft(p, 10, seed=1)]
    assert len(load_sft(p)) == 100


class FakeTok:
    """Токен = символ: декодирование — склейка."""

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class FakeDataset(list):
    def select(self, idx):
        return FakeDataset(self[i] for i in idx)


def row(prompt: str, answer: str) -> dict:
    ids = [ord(c) for c in prompt + answer]
    return {"input_ids": ids, "labels": [-100] * len(prompt) + ids[len(prompt) :]}


def test_mask_check_passes_on_answer_and_reports_share():
    ds = FakeDataset([row("system user", '{"status": "answered"}')])
    assert response_text(ds[0]["input_ids"], ds[0]["labels"], FakeTok()) == '{"status": "answered"}'
    assert 0.6 < check_mask(ds, FakeTok()) < 0.7


def test_mask_check_fails_when_mask_shifted():
    ds = FakeDataset([row("system", 'user {"status": "answered"}')])  # в loss попал вопрос
    with pytest.raises(RuntimeError, match="маска"):
        check_mask(ds, FakeTok())
