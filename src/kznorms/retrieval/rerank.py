"""Реранкер bge-reranker-v2-m3: top-30 → top-5.

Эмбеддер (bi-encoder) кодирует вопрос и пункт по отдельности — быстро, векторы пунктов
считаются заранее, но слова вопроса и пункта друг друга «не видят». Реранкер (cross-encoder)
читает пару «вопрос + пункт» целиком через attention и выдаёт одну оценку релевантности:
точнее, но в 30 раз дороже на кандидата. Поэтому схема двухступенчатая: дешёвый поиск
сужает 800 фрагментов до 30, дорогой реранкер выбирает из них 5.
"""

from kznorms.retrieval.dense import device


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", max_length: int = 512):
        from sentence_transformers import CrossEncoder

        dev = device()
        self.model = CrossEncoder(model_name, max_length=max_length, device=dev)
        if dev == "cuda":
            self.model.model.half()

    def score(self, query: str, texts: list[str], batch_size: int = 16) -> list[float]:
        if not texts:
            return []
        scores = self.model.predict(
            [(query, t) for t in texts], batch_size=batch_size, show_progress_bar=False
        )
        return [float(s) for s in scores]
