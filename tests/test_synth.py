"""Синтетика на подставных учителе и поиске — без GPU и API."""

import json
import random

from kznorms.retrieval.chunking import Chunk
from kznorms.retrieval.retriever import Hit
from kznorms.schemas import Answer, Citation, Clause, SynthExample, SynthQuestion
from kznorms.synth.build_contexts import build_example, jaccard
from kznorms.synth.filter import apply_filter
from kznorms.synth.gen_questions import parse_questions, select_chunks
from kznorms.synth.run import to_sft
from kznorms.synth.teacher import Teacher

CFG = {
    "context_k": 3, "near_dup_jaccard": 0.6, "nf_max_rerank_ratio": 0.8,
    "max_answer_chars": 700, "dedup_jaccard": 0.8,
}  # fmt: skip
GOLD_TEXT = "Ширина марша лестницы должна быть не менее 1,05 м."


def clause(cid: str, text: str) -> Clause:
    return Clause(clause_id=cid, doc_code=cid.split(":")[0], doc_title="Т", number=cid[-3:],
                  text=text)  # fmt: skip


class FakeRetriever:
    def __init__(self) -> None:
        texts = {
            "tr:1.1": GOLD_TEXT,
            "tr:1.2": "Ширина марша лестницы должна быть не менее 1,05 м в жилых домах.",  # дубль
            "tr:1.3": "Высота ограждения лестницы не менее 0,9 м.",
            "tr:1.4": "Уклон марша не более 1:2.",
            "te:9.9": "Тестовый пункт про лестницы.",
            "tr:1.5": "Марши проектируются шириной от 1,05 м.",
        }
        self.clauses = {cid: clause(cid, t) for cid, t in texts.items()}
        self.chunks = {f"{c}#0": Chunk(f"{c}#0", c, f"Т. Раздел\n{t}") for c, t in texts.items()}
        self.clause_chunks = {ch.clause_id: [ch] for ch in self.chunks.values()}
        self.cfg = {"candidates_k": 30}

        # оценки реранкера: tr:1.5 — пересказ gold (слова другие, но отвечает на вопрос)
        self.scores = {"te:9.9": 0.99, "tr:1.1": 0.98, "tr:1.5": 0.95, "tr:1.2": 0.9,
                       "tr:1.3": 0.3, "tr:1.4": 0.2}  # fmt: skip

    def retrieve(self, query, k=None):
        order = ["te:9.9", "tr:1.1", "tr:1.5", "tr:1.2", "tr:1.3", "tr:1.4"]
        return [Hit(self.clauses[c], self.scores[c], self.chunks[f"{c}#0"].text) for c in order]


Q = SynthQuestion(qid="tr:1.1#0/0", split="train", clause_id="tr:1.1", chunk_id="tr:1.1#0",
                  qtype="число", question="Какой ширины делать марш лестницы?")  # fmt: skip
ALLOWED = {"tr"}


def test_answerable_context_has_gold_and_no_test_docs():
    ex = build_example(FakeRetriever(), Q, False, ALLOWED, CFG, random.Random(0))
    ids = [p["clause_id"] for p in ex.passages]
    assert len(ids) == 3 and ids.count("tr:1.1") == 1
    assert "te:9.9" not in ids  # test-документы не попадают даже дистракторами
    assert ex.passages[ids.index("tr:1.1")]["text"] == GOLD_TEXT  # без заголовка фрагмента


def test_gold_position_is_random():
    positions = {
        [p["clause_id"] for p in build_example(
            FakeRetriever(), Q, False, ALLOWED, CFG, random.Random(s)).passages].index("tr:1.1")
        for s in range(20)
    }  # fmt: skip
    assert len(positions) > 1


def test_not_found_context_drops_gold_and_near_duplicates():
    assert jaccard(GOLD_TEXT, FakeRetriever().clauses["tr:1.2"].text) >= 0.6
    ex = build_example(FakeRetriever(), Q, True, ALLOWED, CFG, random.Random(0))
    ids = [p["clause_id"] for p in ex.passages]
    assert ids == ["tr:1.3", "tr:1.4"]  # ни дословного дубля 1.2, ни пересказа 1.5


def test_parse_questions_drops_bad_items():
    raw = json.dumps({"questions": [
        {"type": "число", "question": "Какой ширины делать марш?"},
        {"type": "число", "question": "Ещё один вопрос того же типа?"},
        {"type": "выдуманный", "question": "Вопрос неизвестного типа?"},
        {"type": "можно ли", "question": "Да?"},
    ]}, ensure_ascii=False)  # fmt: skip
    assert parse_questions(raw, ["число", "можно ли"], 3) == [
        ("число", "Какой ширины делать марш?")
    ]
    assert parse_questions("не json", ["число"], 3) == []


def example(not_found: bool, answer: Answer | None, question=Q.question) -> SynthExample:
    q = Q.model_copy(update={"question": question})
    passages = [{"clause_id": "tr:1.1", "doc_code": "tr", "number": "1.1", "text": GOLD_TEXT}]
    return SynthExample(q=q, not_found=not_found, passages=passages, answer=answer)


GOOD = Answer(status="answered", answer="Не менее 1,05 м.",
              citations=[Citation(clause_id="tr:1.1", quote="не менее 1,05 м")])  # fmt: skip


def test_filter_keeps_only_checkable_examples():
    exs = [
        example(False, GOOD),
        example(False, GOOD, question="Какой ширины делать марш лестницы?!"),  # дубль
        example(False, Answer(status="answered", answer="Не менее 1,2 м.",
                              citations=GOOD.citations)),  # число не из цитаты
        example(False, Answer(status="not_found", answer="")),  # отказ при gold
        example(True, GOOD, question="Сколько ступеней в марше?"),  # ответил без gold
        example(True, Answer(status="not_found", answer=""), question="Где перила?"),  # годен
        example(False, None),
    ]  # fmt: skip
    kept = apply_filter(exs, CFG)
    assert [(e.not_found, e.answer.status) for e in kept] == [
        (False, "answered"), (True, "not_found"),
    ]  # fmt: skip
    assert exs[1].reasons == ["дубль вопроса"]
    assert exs[4].reasons == ["учитель нашёл ответ без gold"]


def test_sft_row_is_prod_prompt_plus_answer():
    row = to_sft(example(False, GOOD))
    roles = [m["role"] for m in row["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert "[tr:1.1] tr, п. 1.1" in row["messages"][1]["content"]
    assert Answer.model_validate_json(row["messages"][2]["content"]) == GOOD


def test_select_chunks_skips_test_docs_and_tables():
    r = FakeRetriever()
    r.clause_chunks["tr:1.4"] = [
        Chunk("tr:1.4#0", "tr:1.4", "Т\n1,2 | 3,4 | 5,6 | 7,8 | 9,0 | 1,1")
    ]
    picked = select_chunks(r, {"tr": "train", "te": "test"}, 10, 0.0, 20, 0.5, random.Random(0))
    ids = sorted(ch.clause_id for _, ch in picked)
    assert ids == ["tr:1.1", "tr:1.2", "tr:1.3", "tr:1.5"]


class FakeAPI:
    def __init__(self) -> None:
        self.n = 0

    def _request(self, messages, schema, temperature, max_tokens):
        self.n += 1
        return '{"ok": true}'


def test_teacher_cache(tmp_path):
    t = Teacher("http://x", "m", cache_dir=tmp_path)
    api = FakeAPI()
    t._request = api._request
    msgs = [{"role": "user", "content": "вопрос"}]
    assert t.chat_json(msgs, {}) == '{"ok": true}'
    assert t.chat_json(msgs, {}) == '{"ok": true}'  # второй раз — из кэша
    assert api.n == 1 and t.cached == 1
    t.chat_json(msgs, {}, temperature=0.7)  # другой запрос — новый вызов
    assert api.n == 2


def test_rate_limiter_spaces_calls():
    import time

    from kznorms.synth.teacher import RateLimiter

    lim = RateLimiter(rpm=600)  # 0,1 с между вызовами
    t = time.monotonic()
    for _ in range(4):
        lim.wait()
    assert time.monotonic() - t >= 0.29


def test_teacher_stops_after_consecutive_failures(tmp_path, monkeypatch):
    import pytest

    from kznorms.synth.teacher import QuotaExhausted

    t = Teacher("http://x", "m", cache_dir=tmp_path, retries=1, max_consecutive_failures=2)

    def boom(*a):
        raise ConnectionError("квота")

    t._request = boom
    monkeypatch.setattr("kznorms.synth.teacher.time.sleep", lambda s: None)  # не ждать
    for i in range(2):
        assert t.chat_json([{"role": "user", "content": str(i)}], {}) is None
    with pytest.raises(QuotaExhausted):
        t.chat_json([{"role": "user", "content": "ещё"}], {})


def test_contexts_are_reused_from_cache():
    from kznorms.synth.build_contexts import build_contexts

    cfg = {**CFG, "seed": 1, "not_found_share": 0.5}
    first = build_contexts(FakeRetriever(), [Q], ALLOWED, cfg)

    class Broken(FakeRetriever):
        def retrieve(self, query, k=None):
            raise AssertionError("поиск не должен вызываться — контекст есть в кэше")

    again = build_contexts(Broken(), [Q], ALLOWED, cfg, cache={Q.qid: first[0]})
    assert again[0].passages == first[0].passages
    # тот же вопрос в другом прогоне — то же решение not_found (зерно от qid, не от порядка)
    other = build_contexts(FakeRetriever(), [Q.model_copy(update={"qid": "x/0"}), Q], ALLOWED, cfg)
    assert other[1].not_found == first[0].not_found


def test_daily_quota_stops_immediately(tmp_path, monkeypatch):
    import httpx
    import pytest
    from openai import RateLimitError

    from kznorms.synth.teacher import QuotaExhausted

    monkeypatch.setattr("kznorms.synth.teacher.time.sleep", lambda s: None)
    body = '{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}'
    resp = httpx.Response(429, text=body, request=httpx.Request("POST", "http://x"))
    t = Teacher("http://x", "m", cache_dir=tmp_path)
    calls = []

    def quota(*a):
        calls.append(1)
        raise RateLimitError("quota", response=resp, body=None)

    t._request = quota
    with pytest.raises(QuotaExhausted):
        t.chat_json([{"role": "user", "content": "q"}], {})
    assert len(calls) == 1  # без повторов


def test_stages_run_separately_through_files(tmp_path, monkeypatch):
    import kznorms.synth.run as run_mod

    for name, val in {
        "SYNTHETIC": tmp_path, "SFT": tmp_path / "sft",
        "QUESTIONS": tmp_path / "q.jsonl", "CONTEXTS": tmp_path / "c.jsonl",
    }.items():  # fmt: skip
        monkeypatch.setattr(run_mod, name, val)

    class FakeTeacher:
        model, calls, cached, failed = "fake", 0, 0, 0

        def chat_json(self, messages, schema, temperature=0.0, max_tokens=2048):
            if "questions" in schema.get("properties", {}):
                return json.dumps({"questions": [
                    {"type": "число", "question": "Какой ширины делать марш лестницы?"}]},
                    ensure_ascii=False)  # fmt: skip
            if "tr:1.1" in messages[1]["content"]:
                return GOOD.model_dump_json()
            return Answer(status="not_found", answer="").model_dump_json()

    r = FakeRetriever()
    r.warmup = lambda mode=None: None
    r.clause_chunks = {"tr:1.1": r.clause_chunks["tr:1.1"]}  # единственный кандидат — gold
    cfg = {**CFG, "seed": 1, "not_found_share": 0.0, "n_clauses": 1, "val_share": 0.0,
           "min_passage_chars": 10, "min_letter_share": 0.5, "question_types": ["число"],
           "questions_per_clause": 1, "temperature_questions": 0.7,
           "temperature_answers": 0.0, "workers": 1}  # fmt: skip
    split_of = {"tr": "train", "te": "test"}
    assert run_mod.run(FakeTeacher(), r, split_of, cfg, ("questions",)) is None
    assert (tmp_path / "q.jsonl").exists() and not (tmp_path / "c.jsonl").exists()
    run_mod.run(None, r, split_of, cfg, ("contexts",))  # без учителя
    rep = run_mod.run(FakeTeacher(), r, split_of, cfg, ("answers",))
    assert rep["questions"] == 1 and rep["kept"] == 1
    assert (tmp_path / "sft" / "train.jsonl").read_text(encoding="utf-8").count("\n") == 1


def test_soft_refusal_without_citations_is_relabeled():
    from kznorms.synth.filter import NOT_FOUND_TEXT

    soft = example(True, Answer(status="partial", answer="Не указано в контексте."),
                   question="Где перила?")  # fmt: skip
    cited = example(True, GOOD, question="Сколько ступеней?")  # опора есть — выбрасываем
    gold_soft = example(False, Answer(status="partial", answer="Не хватает данных."))
    kept = apply_filter([soft, cited, gold_soft], CFG)
    assert kept == [soft] and soft.relabeled
    assert soft.answer == Answer(status="not_found", answer=NOT_FOUND_TEXT)
    assert cited.reasons == ["учитель нашёл ответ без gold"]
    assert gold_soft.reasons  # с gold мягкий отказ не исправляем — просто плохой пример
