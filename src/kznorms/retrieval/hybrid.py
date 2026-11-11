"""Reciprocal Rank Fusion результатов BM25 и векторного поиска.

Складывать сырые оценки нельзя: BM25 выдаёт 3…40, косинус 0.3…0.8, шкалы несравнимы и
плавают от запроса к запросу. RRF смотрит только на места в списках:
    score(d) = Σ 1 / (k + rank_i(d))
Документ, который оба поиска поставили высоко, побеждает; найденный только одним тоже
остаётся в списке. k=60 (из статьи Cormack et al., 2009) сглаживает разницу между 1-м и
5-м местом, чтобы один поиск не перетягивал всё на себя.
"""

from collections import defaultdict


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def collapse(hits: list[tuple[str, float]], chunk_to_clause: dict[str, str]) -> list[str]:
    """Ранжированные фрагменты → ранжированные пункты (место пункта = место его лучшего
    фрагмента). Иначе длинный пункт из 8 фрагментов займёт половину выдачи."""
    seen: dict[str, None] = {}
    for chunk_id, _ in hits:
        seen.setdefault(chunk_to_clause[chunk_id], None)
    return list(seen)
