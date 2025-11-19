from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np

from doc_studio.retrieval.interfaces import ScoredDocument  # adjust import path as needed


def _l2n(X: np.ndarray) -> np.ndarray:
    """L2-normalize each row."""
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / n


class MMR:
    """
    Maximal Marginal Relevance reranker.
    Reranks docs by a relevance–diversity tradeoff:

        score = λ * sim(doc, query) - (1 - λ) * max_sim(doc, already_selected)

    where sims are cosine similarities in embedding space.
    """

    def __init__(self, embedding_fn: Callable[[list[str]], np.ndarray], lam: float = 0.6) -> None:
        self.f = embedding_fn
        self.lam = lam

    def rerank(self, query: str, cands: List[ScoredDocument], k: int) -> List[ScoredDocument]:
        """
        Rerank candidates using MMR.

        Args:
            query: user query string
            cands: list of ScoredDocument from the base retriever
            k: number of documents to return

        Returns:
            List[ScoredDocument] with updated `score` = MMR score and
            `meta` including mmr-specific metadata.
        """
        if not cands:
            return []

        k = min(k, len(cands))

        # Embed query and candidate texts
        texts = [c.doc.text for c in cands]
        Q = _l2n(self.f([query]))         # shape (1, d)
        D = _l2n(self.f(texts))           # shape (n, d)

        # Base relevance: cosine sim(query, doc)
        rel = (D @ Q.T).ravel()           # shape (n,)

        selected: List[int] = []
        remaining: List[int] = list(range(len(cands)))

        # store final MMR scores per index
        mmr_scores = np.zeros(len(cands), dtype=np.float32)

        while remaining and len(selected) < k:
            if not selected:
                # First pick: highest relevance
                rem_rel = rel[remaining]
                j_rel = int(np.argmax(rem_rel))
                chosen_idx = remaining.pop(j_rel)
                selected.append(chosen_idx)
                mmr_scores[chosen_idx] = float(rel[chosen_idx])
            else:
                # Subsequent picks: balance relevance vs diversity
                S = D[selected]                 # already selected
                sims = D[remaining] @ S.T       # (len(remaining), len(selected))
                mx = sims.max(axis=1)           # max similarity to any selected
                score = self.lam * rel[remaining] - (1.0 - self.lam) * mx

                j_score = int(np.argmax(score))
                chosen_idx = remaining.pop(j_score)
                selected.append(chosen_idx)
                mmr_scores[chosen_idx] = float(score[j_score])

        # Build new ScoredDocument list with metadata
        out: List[ScoredDocument] = []
        for rank, i in enumerate(selected):
            base_sd = cands[i]
            base_meta = base_sd.meta or {}

            out_meta = {
                **base_meta,
                "mmr_score": float(mmr_scores[i]),
                "base_score": float(rel[i]),              # original cosine sim
                "original_rank": base_meta.get("rank", i),
                "rank": rank,                             # new rank after MMR
                "reranked_from": base_meta.get("retriever", "unknown"),
            }

            out.append(
                ScoredDocument(
                    doc=base_sd.doc,
                    score=float(mmr_scores[i]),           # now MMR score
                    meta=out_meta,
                )
            )

        return out
