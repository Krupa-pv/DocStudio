from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Tuple

from doc_studio.core.llm import LLMClient
from doc_studio.core.types import ChatMessage, Role, ChatResult
from doc_studio.retrieval import ScoredDocument


@dataclass
class AnalystScores:
    faithfulness: float    # 0..1 – are claims supported by provided context?
    coverage: float        # 0..1 – does it answer the query and use key info?
    clarity: float         # 0..1 – organization/grammar/citations
    overall: float         # 0..1 – weighted blend


@dataclass
class CritiqueResult:
    scores: AnalystScores
    justification: str
    suggested_rewrite: str | None
    llm: ChatResult


class AnalystAgent:
    """
    Grades a summary against retrieved context and original query.
    Produces numeric scores in [0,1], a short justification, and (optionally) a rewritten summary.
    """

    _SYSTEM = (
        "You are a strict technical reviewer. Score between 0 and 1 with two decimals. "
        "Be conservative. Prefer truth and traceability over style."
    )

    def __init__(
        self,
        llm: LLMClient,
        *,
        do_rewrite: bool = True,
        weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),  # faithfulness, coverage, clarity
    ) -> None:
        self.llm = llm
        self.do_rewrite = do_rewrite
        self.w_f, self.w_c, self.w_cl = weights

    def _pack_context(self, hits: List[ScoredDocument]) -> str:
        pieces: List[str] = []
        for h in hits:
            body = h.doc.text.strip().replace("\n", " ")
            pieces.append(f"[{h.doc.doc_id}] {body}")
        return "\n".join(pieces)

    def critique(
        self,
        query: str,
        hits: List[ScoredDocument],
        summary: str,
        temperature: float = 0.0,
        max_tokens: int = 400,
        context_override: str | None = None, 
    ) -> CritiqueResult:
        ctx = context_override or self._pack_context(hits)

        # rewrite line; avoids backslashes in f-string expr
        rewrite_line = '"rewrite": "string"' if self.do_rewrite else '"rewrite": null'

        # user prompt; strict json
        user = (
            "You will evaluate a SUMMARY against a QUERY and CONTEXT (ground-truth snippets with IDs like [d4]).\n"
            "1) Score faithfulness: are all claims supported by the provided CONTEXT? Penalize any claim that cannot be grounded.\n"
            "2) Score coverage: does the SUMMARY answer the QUERY and include the key ideas from CONTEXT?\n"
            "3) Score clarity: organization, concision, correct inline citations like [d4][d2].\n"
            "4) Provide a 1-3 sentence justification.\n"
            f"5) {'Provide a rewritten summary that improves those issues and keeps citations.' if self.do_rewrite else 'Do not rewrite.'}\n\n"
            "Return STRICT JSON ONLY with this schema (no extra text):\n"
            "{\n"
            '  "faithfulness": 0.00,\n'
            '  "coverage": 0.00,\n'
            '  "clarity": 0.00,\n'
            '  "justification": "string",\n'
            f"  {rewrite_line}\n"
            "}\n\n"
            f"QUERY:\n{query}\n\n"
            f"CONTEXT:\n{ctx}\n\n"
            f"SUMMARY:\n{summary}\n"
        )

        msgs = [
            ChatMessage(role=Role.SYSTEM, content=self._SYSTEM),
            ChatMessage(role=Role.USER, content=user),
        ]
        res = self.llm.chat(messages=msgs, temperature=temperature, max_tokens=max_tokens)

        # JSON parse 
        data = {}
        try:
            data = json.loads(res.text)
        except json.JSONDecodeError:
            
            try:
                start = res.text.find("{")
                end = res.text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    data = json.loads(res.text[start : end + 1])
            except Exception:
                data = {}

        faith = float(data.get("faithfulness", 0.0))
        cov = float(data.get("coverage", 0.0))
        clar = float(data.get("clarity", 0.0))
        overall = max(0.0, min(1.0, self.w_f * faith + self.w_c * cov + self.w_cl * clar))

        scores = AnalystScores(
            faithfulness=faith,
            coverage=cov,
            clarity=clar,
            overall=overall,
        )
        rewrite = data.get("rewrite") if self.do_rewrite else None

        return CritiqueResult(
            scores=scores,
            justification=str(data.get("justification", "")).strip(),
            suggested_rewrite=(str(rewrite).strip() if rewrite else None),
            llm=res,
        )
