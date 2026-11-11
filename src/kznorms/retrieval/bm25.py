"""BM25 (bm25s) с лемматизацией pymorphy3.

BM25 ищет по совпадению слов. В русском у слова десятки форм («лестница», «лестницы»,
«лестничной»), поэтому без нормализации «ширина лестницы» не найдёт «ширину лестниц».
Стемминг (отрезание окончаний) грубый, лемматизация pymorphy3 приводит слово к словарной
форме: «лестниц» → «лестница». Одинаковая функция tokenize() применяется и к корпусу,
и к запросу — иначе токены не совпадут.
"""

import json
import re
from functools import lru_cache
from pathlib import Path

import bm25s
import pymorphy3

from kznorms.retrieval.chunking import Chunk

# Слова (кириллица/латиница) и числа с десятичными/пунктными точками: «1,2», «5.3.2».
_TOKEN = re.compile(r"[а-яa-z]+|\d+(?:[.,]\d+)*")

_STOP = frozenset(bm25s.stopwords.STOPWORDS_RUSSIAN) | {"также", "либо", "который", "этот"}


@lru_cache(maxsize=1)
def _morph() -> pymorphy3.MorphAnalyzer:
    return pymorphy3.MorphAnalyzer()


@lru_cache(maxsize=200_000)
def lemma(word: str) -> str:
    # parse() возвращает варианты разбора по убыванию вероятности; берём первый.
    # Омонимия («стали» — сталь/стать) иногда ошибается, для поиска это терпимо.
    return _morph().parse(word)[0].normal_form.replace("ё", "е")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for m in _TOKEN.finditer(text.lower().replace("ё", "е")):
        w = m.group()
        if w[0].isdigit():
            tokens.append(w.replace(",", "."))  # «1,5» и «1.5» — одно число
            continue
        if w in _STOP:
            continue
        lw = lemma(w)
        if lw not in _STOP:
            tokens.append(lw)
    return tokens


class BM25Index:
    def __init__(self, bm25: bm25s.BM25, chunk_ids: list[str]):
        self.bm25 = bm25
        self.chunk_ids = chunk_ids

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "BM25Index":
        bm25 = bm25s.BM25()  # Lucene-вариант, k1=1.5, b=0.75
        bm25.index([tokenize(c.text) for c in chunks], show_progress=False)
        return cls(bm25, [c.chunk_id for c in chunks])

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.bm25.save(str(path), show_progress=False)
        (path / "chunk_ids.json").write_text(json.dumps(self.chunk_ids), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        bm25 = bm25s.BM25.load(str(path), show_progress=False)
        ids = json.loads((path / "chunk_ids.json").read_text(encoding="utf-8"))
        return cls(bm25, ids)

    def search(self, query: str, k: int = 30) -> list[tuple[str, float]]:
        """→ [(chunk_id, score)] по убыванию; пусто, если ни одно слово не встретилось."""
        q = [t for t in tokenize(query) if t in self.bm25.vocab_dict]
        if not q:
            return []
        k = min(k, len(self.chunk_ids))
        docs, scores = self.bm25.retrieve([q], k=k, show_progress=False)
        return [
            (self.chunk_ids[int(d)], float(s))
            for d, s in zip(docs[0], scores[0], strict=True)
            if s > 0
        ]
