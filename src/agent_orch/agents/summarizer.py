from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from agent_orch.core.llm import LLMClient
from agent_orch.core.types import ChatMessage, Role, ChatResult
from agent_orch.retrieval import Retriever, ScoredDocument


@dataclass
class SummaryResult:
    summary: str
    used_docs: List[Tuple[str, float]]  #(doc_id, score)
    llm: ChatResult


class SummarizerAgent:
    """
    RAG summarizer:
      1)retrieve top-k docs
      2)pack context into prompt (with [doc_id] headers)
      3)ask model for concise summary with inline citations like [d4]
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        *,
        k: int = 5,
        context_budget_chars: int = 3000,
        system_prompt: str | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.k = k
        self.context_budget_chars = context_budget_chars
        self.system_prompt = system_prompt or (
            "You are a technical summarizer. Write concise, factual summaries with inline citations "
            "using the provided document IDs in square brackets, e.g., [d4][d2]. Do not invent citations."
        )

    def _pack_context(self, hits: List[ScoredDocument]) -> str:
        """Pack top-k docs into context block thats boundeds."""
        pieces: List[str] = []
        total = 0
        for h in hits[: self.k]:
            header = f"[{h.doc.doc_id}] (score={h.score:.3f})"
            body = h.doc.text.strip().replace("\n", " ")
            chunk = f"{header}\n{body}\n"
            if total + len(chunk) > self.context_budget_chars:
                break
            pieces.append(chunk)
            total += len(chunk)
        return "\n".join(pieces)

    def summarize(self, query: str, temperature: float = 0.2, max_tokens: int = 400) -> SummaryResult:
        # 1- retrieve
        hits = self.retriever.search(query, k=self.k)

        #2-pack context
        context = self._pack_context(hits)

        #3 -prompt
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

        # 4 - call model
        llm_res = self.llm.chat(messages=msgs, temperature=temperature, max_tokens=max_tokens)

        return SummaryResult(
            summary=llm_res.text,
            used_docs=[(h.doc.doc_id, h.score) for h in hits[: self.k]],
            llm=llm_res,
        )
