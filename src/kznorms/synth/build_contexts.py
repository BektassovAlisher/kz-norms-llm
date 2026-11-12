"""Контексты настоящим ретривером; доля примеров без gold-пункта → not_found.

Контекст строим тем же поиском, что в проде: дистракторы получаются «трудные» — пункты на
ту же тему. Три правила против утечек и перекосов:
1. Поиск только по train/val-документам — иначе test-пункты попадут в обучение дистракторами.
2. Gold ставим в случайную позицию — иначе модель выучит «ответ в первом пункте»
   (в eval базовая модель заметно хуже, когда gold не первый).
3. В not_found-примерах убираем не только gold, но и всё, что может ответить на вопрос:
   СН и СП часто повторяют одно правило, и «отказ» при дубле в контексте учил бы врать.

Как искать «то, что может ответить»: сходство текстов не годится. СН↔СП — одно правило
пересказом: совпадение слов 0.10, косинус BGE-M3 0.73 — а соседние пункты одной темы с
разными требованиями дают косинус 0.88. Эмбеддинг меряет «о том же ли», а нужно «отвечает
ли на этот вопрос» — это вопрос к реранкеру (кросс-энкодер видит пару вопрос+пункт).
На пилоте: дистрактор с оценкой < 0.6 × оценки gold — учитель всегда говорил not_found,
≥ 0.9 × — почти всегда находил ответ. Порог `nf_max_rerank_ratio` = 0.8.
"""

import random

from kznorms.rag.pipeline import to_passage
from kznorms.rag.prompts import Passage
from kznorms.rag.verify import normalize
from kznorms.schemas import SynthExample, SynthQuestion
from kznorms.splits import doc_of
from kznorms.synth.gen_questions import body


def jaccard(a: str, b: str) -> float:
    wa, wb = set(normalize(a).split()), set(normalize(b).split())
    return len(wa & wb) / max(len(wa | wb), 1)


def build_example(
    retriever, q: SynthQuestion, not_found: bool, allowed_docs: set[str], cfg: dict,
    rng: random.Random,
) -> SynthExample:  # fmt: skip
    k = cfg["context_k"]
    gold_chunk = retriever.chunks[q.chunk_id]
    gold_clause = retriever.clauses[q.clause_id]
    gold = Passage(q.clause_id, gold_clause.doc_code, gold_clause.number, body(gold_chunk.text))

    found = retriever.retrieve(q.question, k=retriever.cfg["candidates_k"])
    hits = [
        h
        for h in found
        if doc_of(h.clause.clause_id) in allowed_docs and h.clause.clause_id != q.clause_id
    ]
    if not_found:
        # оценка gold тем же реранкером: из выдачи, если gold нашёлся, иначе отдельно
        gold_score = next((h.score for h in found if h.clause.clause_id == q.clause_id), None)
        if gold_score is None:
            gold_score = float(retriever.reranker.score(q.question, [gold_chunk.text])[0])
        limit = cfg["nf_max_rerank_ratio"] * gold_score
        hits = [
            h
            for h in hits
            if h.score < limit and jaccard(body(h.chunk), gold.text) < cfg["near_dup_jaccard"]
        ]
    passages = [to_passage(h) for h in hits[: k if not_found else k - 1]]
    if not not_found:
        passages.insert(rng.randrange(len(passages) + 1), gold)
    return SynthExample(q=q, not_found=not_found, passages=[p.__dict__ for p in passages])


def build_contexts(
    retriever, questions, allowed_docs, cfg, cache: dict[str, SynthExample] | None = None
) -> list[SynthExample]:
    """Контекст для каждого вопроса; готовые (из прошлого запуска) берутся из cache.

    Случайность — своя на каждый вопрос (зерно = seed + qid): тогда «not_found или нет» и
    позиция gold не зависят от того, какие ещё вопросы в прогоне, и кэш остаётся верным.
    """
    from rich.progress import track

    cache = cache or {}
    out = []
    for q in track(questions, description="контексты"):
        if q.qid in cache and cache[q.qid].q == q:
            out.append(cache[q.qid].model_copy(update={"answer": None, "reasons": []}))
            continue
        rng = random.Random(f"{cfg['seed']}:{q.qid}")
        not_found = rng.random() < cfg["not_found_share"]
        out.append(build_example(retriever, q, not_found, allowed_docs, cfg, rng))
    return out
