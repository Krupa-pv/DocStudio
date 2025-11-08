from __future__ import annotations

from typing import Iterable, List
import numpy as np

from .interfaces import Document, ScoredDocument, Retriever
from .hf_embedder import HFEmbedder


class HFEmbeddingsRetriever(Retriever):
    """
    in-memory dense retriever (cosine via dot on L2-normalized vectors)
    """
    def __init__(self, embedder: HFEmbedder | None = None):
        self.embedder = embedder or HFEmbedder()
        self._docs: list[Document] = []
        self._embs: np.ndarray | None = None  #shape: (N, D), L2-normalized

    def add(self, docs: Iterable[Document]) -> None:
        new_docs = list(docs)
        if not new_docs:
            return
        self._docs.extend(new_docs)
        corpus = [d.text for d in self._docs]
        self._embs = self.embedder(corpus)  # normalized

    def search(self, query: str, k: int = 5) -> List[ScoredDocument]:
        if not self._docs or self._embs is None:
            return []
        q = self.embedder([query])  # (1, D), normalized
        sims = (self._embs @ q.T).ravel()   #cosine similarity
        idx = np.argsort(-sims)[:k]
        return [ScoredDocument(self._docs[i], float(sims[i])) for i in idx]

    def size(self) -> int:
        return len(self._docs)
