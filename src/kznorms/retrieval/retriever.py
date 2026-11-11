"""Единая точка входа retrieve(query, k) с режимом из configs/retrieval.yaml.

Режимы нужны для абляции в отчёте: bm25 → dense → hybrid → hybrid_rerank, и на каждом
шаге видно, сколько Recall@5 добавила ступень и сколько миллисекунд она стоит.
Модели грузятся лениво: режим bm25 не тянет torch и работает за миллисекунды.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kznorms.config import Settings, load_yaml
from kznorms.paths import BM25_DIR, CHUNKS_JSONL, CLAUSES_JSONL, ROOT
from kznorms.retrieval.bm25 import BM25Index
from kznorms.retrieval.chunking import Chunk, make_chunks
from kznorms.retrieval.hybrid import collapse, rrf
from kznorms.schemas import Clause

MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")


@dataclass
class Hit:
    clause: Clause
    score: float
    chunk: str  # лучший фрагмент пункта: для длинных пунктов в промпт пойдёт он, а не 10k символов


def load_clauses(path: Path = CLAUSES_JSONL) -> list[Clause]:
    return [Clause.model_validate_json(x) for x in path.read_text(encoding="utf-8").splitlines()]


def save_chunks(chunks: list[Chunk], path: Path = CHUNKS_JSONL) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")


def load_chunks(path: Path = CHUNKS_JSONL) -> list[Chunk]:
    return [Chunk(**json.loads(x)) for x in path.read_text(encoding="utf-8").splitlines()]


def build_chunks(cfg: dict[str, Any], clauses: list[Clause]) -> list[Chunk]:
    return make_chunks(clauses, cfg["chunk_max_chars"], cfg["chunk_overlap"])


def open_dense(cfg: dict[str, Any]):
    from kznorms.retrieval.dense import DenseIndex, Embedder

    path = ROOT / cfg["qdrant_path"] if cfg.get("qdrant_path") else None
    return DenseIndex(
        Embedder(cfg["embedder"], cfg["max_seq_length"]),
        collection=cfg["collection"],
        path=path,
        url=Settings().qdrant_url,
    )


class Retriever:
    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = cfg or load_yaml("retrieval")
        if not CHUNKS_JSONL.exists():
            raise FileNotFoundError("Индекс не построен: запусти `uv run kznorms index`")
        if CLAUSES_JSONL.stat().st_mtime > CHUNKS_JSONL.stat().st_mtime:
            print("Внимание: clauses.jsonl новее индекса — перестрой `uv run kznorms index`")
        self.clauses = {c.clause_id: c for c in load_clauses()}
        self.chunks = {c.chunk_id: c for c in load_chunks()}
        self.chunk_to_clause = {c.chunk_id: c.clause_id for c in self.chunks.values()}
        self.clause_chunks: dict[str, list[Chunk]] = {}
        for c in self.chunks.values():
            self.clause_chunks.setdefault(c.clause_id, []).append(c)
        self._bm25: BM25Index | None = None
        self._dense = None
        self._reranker = None
        self.timings_ms: dict[str, float] = {}

    # --- ленивые компоненты ---
    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index.load(BM25_DIR)
        return self._bm25

    @property
    def dense(self):
        if self._dense is None:
            self._dense = open_dense(self.cfg)
        return self._dense

    @property
    def reranker(self):
        if self._reranker is None:
            from kznorms.retrieval.rerank import Reranker

            self._reranker = Reranker(self.cfg["reranker"], self.cfg["max_seq_length"])
        return self._reranker

    def warmup(self, mode: str | None = None) -> None:
        """Загрузить модели и сделать холостой прогон: первый вызов на CUDA в разы медленнее
        (инициализация ядер), и без прогрева тайминги первого запроса врут."""
        mode = mode or self.cfg["mode"]
        self.bm25.search("прогрев", 1)  # заодно грузит словари pymorphy3
        if mode != "bm25":
            self.dense.embedder.encode(["прогрев"])
        if mode == "hybrid_rerank":
            self.reranker.score("прогрев", ["прогрев"])

    # --- поиск ---
    def _timed(self, name: str, fn, *args):
        t = time.perf_counter()
        out = fn(*args)
        self.timings_ms[name] = (time.perf_counter() - t) * 1000
        return out

    def _best_chunk(self, clause_id: str, hits: list[tuple[str, float]]) -> str:
        for chunk_id, _ in hits:
            if self.chunk_to_clause[chunk_id] == clause_id:
                return self.chunks[chunk_id].text
        return self.clause_chunks[clause_id][0].text

    def retrieve(self, query: str, k: int | None = None, mode: str | None = None) -> list[Hit]:
        mode = mode or self.cfg["mode"]
        if mode not in MODES:
            raise ValueError(f"mode должен быть одним из {MODES}, получено {mode!r}")
        k = k or self.cfg["final_k"]
        n_cand = self.cfg["candidates_k"]
        fetch = n_cand * 3  # фрагментов больше, чем пунктов: берём с запасом до схлопывания
        self.timings_ms = {}

        chunk_hits: list[tuple[str, float]] = []
        rankings: list[list[str]] = []
        if mode in ("bm25", "hybrid", "hybrid_rerank"):
            bm = self._timed("bm25", self.bm25.search, query, fetch)
            chunk_hits += bm
            rankings.append(collapse(bm, self.chunk_to_clause)[:n_cand])
        if mode in ("dense", "hybrid", "hybrid_rerank"):
            de = self._timed("dense", self.dense.search, query, fetch)
            chunk_hits += de
            rankings.append(collapse(de, self.chunk_to_clause)[:n_cand])

        if mode in ("bm25", "dense"):
            scores = {}
            for chunk_id, s in chunk_hits:
                scores.setdefault(self.chunk_to_clause[chunk_id], s)
            ranked = list(scores.items())[:k]
        else:
            ranked = rrf(rankings, self.cfg["rrf_k"])[: n_cand if mode == "hybrid_rerank" else k]

        if mode != "hybrid_rerank":
            return [
                Hit(self.clauses[cid], s, self._best_chunk(cid, chunk_hits)) for cid, s in ranked
            ]

        # Реранкер оценивает каждый фрагмент кандидата; оценка пункта — максимум по фрагментам.
        cand = [ch for cid, _ in ranked for ch in self.clause_chunks[cid]]
        scores = self._timed("rerank", self.reranker.score, query, [c.text for c in cand])
        best: dict[str, tuple[float, str]] = {}
        for ch, s in zip(cand, scores, strict=True):
            if ch.clause_id not in best or s > best[ch.clause_id][0]:
                best[ch.clause_id] = (s, ch.text)
        top = sorted(best.items(), key=lambda x: x[1][0], reverse=True)[:k]
        return [Hit(self.clauses[cid], s, text) for cid, (s, text) in top]
