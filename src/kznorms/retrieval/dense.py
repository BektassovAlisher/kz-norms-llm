"""Эмбеддинги BGE-M3 и коллекция в Qdrant.

Векторный поиск ловит смысл, а не слова: «можно ли не делать лифт в пятиэтажке» найдёт
пункт про «оборудование лифтами жилых зданий высотой ...», хотя общих слов почти нет.
BGE-M3 — многоязычная модель (русский и казахский), вектор 1024, косинусная близость.
Префикс-инструкция к запросу ей не нужна (в отличие от e5/bge-v1.5).

Qdrant работает во встроенном режиме (файлы в artifacts/), без сервера и Docker. Тот же
клиент умеет ходить на сервер — достаточно задать url вместо path.
"""

from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from kznorms.retrieval.chunking import Chunk


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-m3", max_seq_length: int = 512):
        from sentence_transformers import SentenceTransformer

        dev = device()
        self.model = SentenceTransformer(model_name, device=dev)
        if dev == "cuda":
            self.model.half()  # fp16: вдвое меньше VRAM, на качество поиска не влияет
        self.model.max_seq_length = max_seq_length

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,  # единичные векторы: косинус = скалярное произведение
            convert_to_numpy=True,
            show_progress_bar=len(texts) > batch_size,
        )

    def count_truncated(self, texts: list[str]) -> int:
        """Сколько текстов длиннее max_seq_length токенов — их хвост эмбеддер не увидит."""
        tok = self.model.tokenizer
        lens = [len(ids) for ids in tok(texts, add_special_tokens=True)["input_ids"]]
        return sum(n > self.model.max_seq_length for n in lens)


class DenseIndex:
    def __init__(
        self,
        embedder: Embedder,
        collection: str = "clauses",
        path: Path | None = None,
        url: str = "",
    ):
        self.embedder = embedder
        self.collection = collection
        self.client = QdrantClient(path=str(path)) if path else QdrantClient(url=url)

    def build(self, chunks: list[Chunk], batch_size: int = 16) -> None:
        vecs = self.embedder.encode([c.text for c in chunks], batch_size=batch_size)
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=vecs.shape[1], distance=models.Distance.COSINE),
        )
        self.client.upload_points(
            self.collection,
            points=[
                models.PointStruct(
                    id=i,
                    vector=v.tolist(),
                    payload={"chunk_id": c.chunk_id, "clause_id": c.clause_id},
                )
                for i, (c, v) in enumerate(zip(chunks, vecs, strict=True))
            ],
        )

    def search(self, query: str, k: int = 30) -> list[tuple[str, float]]:
        """→ [(chunk_id, cosine)] по убыванию."""
        vec = self.embedder.encode([query])[0]
        res = self.client.query_points(
            self.collection, query=vec.tolist(), limit=k, with_payload=True
        )
        return [(p.payload["chunk_id"], float(p.score)) for p in res.points]

    def close(self) -> None:
        self.client.close()
