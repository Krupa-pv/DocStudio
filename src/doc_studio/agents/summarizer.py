from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Callable, Optional

from doc_studio.core.llm import LLMClient
from doc_studio.core.types import ChatMessage, Role, ChatResult
from doc_studio.retrieval import Retriever, ScoredDocument
from doc_studio.retrieval.mmr import MMR


@dataclass
class SummaryResult:
    summary: str
    used_docs: List[Tuple[str, float]]  #(doc_id, score)
    llm: ChatResult
    context_str: str | None = None


class SummarizerAgent:
    """
    RAG summarizer:
      1)retrieve top-k docs from FAISS
      2) mmr rerank 
      3) pack context into prompt (with [doc_id] headers)
      4) ask llm for concise answers with inline citatinos 
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        *,
        k: int = 5,
        context_budget_chars: int = 3000,
        system_prompt: str | None = None,
        use_mmr: bool = False,  # mmr is optional  
        mmr_lambda: float = 0.6, # relevance vs diversity 
        embedding_fn: Optional[Callable[[list[str]], "np.ndarray"]] = None, #needed if mmr used
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.k = k
        self.context_budget_chars = context_budget_chars
        self.system_prompt = system_prompt or (
            "You are a technical summarizer. Write concise, factual summaries with inline citations "
            "using the provided document IDs in square brackets, e.g., [d4][d2]. Do not invent citations."
        )
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda
        self.embedding_fn = embedding_fn

    def _mmr_pick(self, query: str, base_hits: List[ScoredDocument]) -> List[ScoredDocument]:
        # mmr: balance relevance + diversity, cut down to k; needs embeddings
        if not (self.use_mmr and self.embedding_fn and base_hits):
            return base_hits[: self.k]
        reranker = MMR(embedding_fn=self.embedding_fn, lam=self.mmr_lambda)
        return reranker.rerank(query, base_hits, k=self.k)
    
    def _pack_context(self, hits: List[ScoredDocument]) -> str:
        """Pack top-k docs into context block with budget limit"""
        pieces: List[str] = []
        total = 0
        for h in hits[: self.k]:
            meta_str = ""
            if h.doc.meta:
                meta_str = " | " + ", ".join(f"{k}: {v}" for k, v in h.doc.meta.items())

            header = f"[{h.doc.doc_id}]{meta_str} (score={h.score:.3f})"
            body = h.doc.text.strip().replace("\n", " ")
            chunk = f"{header}\n{body}\n"
            if total + len(chunk) > self.context_budget_chars:
                break
            pieces.append(chunk)
            total += len(chunk)
        return "\n".join(pieces)

    def summarize(self, query: str, temperature: float = 0.2, max_tokens: int = 400) -> SummaryResult:
        # 1- retrieve bit more than k so mmr has room to diversify
        raw_k = max(self.k * 2, self.k)  # small oversample; tweak as needed
        base_hits = self.retriever.search(query, k=raw_k)

        #2-optional mmr rerank down to k
        hits = self._mmr_pick(query, base_hits)
        # 3 - pack bounded context 

        #3 -prompt
        context = self._pack_context(hits)

        # 4) prompt model
        user_prompt = (
            "Query:\n"
            f"{query}\n\n"
            "Context (each item is a document with an ID):\n"
            f"{context}\n\n"
            "Task: Produce a concise summary (5-8 sentences) that answers the query.\n"
            "Use inline citations by doc_id like [d4][d2]. If information is missing, say so.\n"
        )
        msgs = [
            ChatMessage(role=Role.SYSTEM, content=self.system_prompt),
            ChatMessage(role=Role.USER, content=user_prompt),
        ]
        llm_res = self.llm.chat(messages=msgs, temperature=temperature, max_tokens=max_tokens)

        return SummaryResult(
            summary=llm_res.text,
            used_docs=[(h.doc.doc_id, h.score) for h in hits],
            llm=llm_res,
            context_str=context,
        )
    