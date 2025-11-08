from __future__ import annotations
import csv, os, time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from agent_orch.core.llm import LLMClient
from agent_orch.core.types import ChatMessage, Role
from agent_orch.retrieval import Retriever, Document, ScoredDocument
from agent_orch.agents.summarizer import SummarizerAgent, SummaryResult  # <-- new mmr version
from agent_orch.agents.analyst import AnalystAgent, CritiqueResult


@dataclass
class OrchestratorConfig:
    quality_threshold: float = 0.75
    max_attempts: int = 2
    use_rewrite_on_retry: bool = True
    reward_alpha_tokens: float = 0.02   # per 1k tokens
    reward_beta_latency: float = 0.02   # per second


@dataclass
class OrchestratorResult:
    query: str
    attempts: int
    final_summary: str
    final_scores: Dict[str, float]
    used_docs: List[str]
    timeline: List[Dict[str, Any]]


class Orchestrator:
    """
    simple orchestrator using FAISS+MMR retriever and analyst critic
      - summarize -> critique -> maybe retry (rewrite or resummarize)
      - log the numeric reward to csv for later bandit tuning
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        emb_fn=None,                     # new: pass HFEmbedder for MMR
        cfg: Optional[OrchestratorConfig] = None,
        reward_log_path: str = "data/rewards.csv",
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.emb_fn = emb_fn
        self.cfg = cfg or OrchestratorConfig()
        self.reward_log_path = reward_log_path

        self._base_k = 5
        self._base_context_chars = 3000
        self._base_system_prompt = (
            "You are a technical summarizer. Write concise, factual summaries with inline citations "
            "using the provided document IDs in square brackets, e.g., [d4][d2]. Do not invent citations."
        )

        # default summarizer now includes MMR support
        self._summarizer = SummarizerAgent(
            retriever=self.retriever,
            llm=self.llm,
            k=self._base_k,
            context_budget_chars=self._base_context_chars,
            use_mmr=True,
            mmr_lambda=0.6,
            embedding_fn=self.emb_fn,
        )
        self._analyst = AnalystAgent(llm=self.llm, do_rewrite=True)

        # ensure reward csv header exists
        os.makedirs(os.path.dirname(self.reward_log_path), exist_ok=True)
        if not os.path.exists(self.reward_log_path):
            with open(self.reward_log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "ts","query","arm","reward","overall","faithfulness","coverage","clarity",
                    "total_tokens","latency_ms","attempts"
                ])

    def _build_summarizer_from_arm(self, arm_params: dict) -> SummarizerAgent:
        # clones summarizer w/ tweaks per run
        return SummarizerAgent(
            retriever=self.retriever,
            llm=self.llm,
            k=arm_params.get("k", self._base_k),
            context_budget_chars=arm_params.get("context_budget_chars", self._base_context_chars),
            system_prompt=arm_params.get("system_prompt", self._base_system_prompt),
            use_mmr=arm_params.get("use_mmr", True),
            mmr_lambda=arm_params.get("mmr_lambda", 0.6),
            embedding_fn=self.emb_fn,
        )

    def _compute_reward(self, crit: CritiqueResult) -> float:
        # same reward function as before, using real token+latency fields
        toks_pen = self.cfg.reward_alpha_tokens * (crit.llm.total_tokens / 1000.0)
        lat_pen = self.cfg.reward_beta_latency * (crit.llm.latency_ms / 1000.0)
        return max(0.0, crit.scores.overall - toks_pen - lat_pen)

    def _log_reward(self, query: str, arm_name: str, reward: float, crit: CritiqueResult, attempts: int) -> None:
        with open(self.reward_log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                int(time.time()),
                query,
                arm_name,
                round(reward, 4),
                round(crit.scores.overall, 4),
                round(crit.scores.faithfulness, 4),
                round(crit.scores.coverage, 4),
                round(crit.scores.clarity, 4),
                crit.llm.total_tokens,
                round(crit.llm.latency_ms, 2),
                attempts,
            ])

    def run(self, query: str, arm_name: str = "default", arm_params: dict | None = None) -> OrchestratorResult:
        arm_params = arm_params or {}
        timeline: List[Dict[str, Any]] = []
        summarizer = self._build_summarizer_from_arm(arm_params)

        # ---- attempt 1 ----
        sum_res: SummaryResult = summarizer.summarize(
            query,
            temperature=arm_params.get("temperature", 0.0),
            max_tokens=arm_params.get("max_tokens", 300)
        )
        hits = self.retriever.search(query, k=arm_params.get("k", self._base_k))
        crit1: CritiqueResult = self._analyst.critique(query=query, hits=hits, summary=sum_res.summary, context_override=sum_res.context_str)

        timeline.append({
            "step": "attempt_1",
            "used_docs": [d.doc.doc_id for d in hits],
            "summary": sum_res.summary,
            "scores": {
                "overall": crit1.scores.overall,
                "faithfulness": crit1.scores.faithfulness,
                "coverage": crit1.scores.coverage,
                "clarity": crit1.scores.clarity,
            },
            "critic_latency_ms": crit1.llm.latency_ms,
            "critic_tokens": crit1.llm.total_tokens,
            "rewrite": crit1.suggested_rewrite,
            "justification": crit1.justification,
        })

        final_sum = sum_res.summary
        final_crit = crit1
        attempts = 1

        # ---- optional retry ----
        if crit1.scores.overall < self.cfg.quality_threshold and attempts < self.cfg.max_attempts:
            attempts += 1
            if self.cfg.use_rewrite_on_retry and crit1.suggested_rewrite:
                retry_summary = crit1.suggested_rewrite
            else:
                nudged = SummarizerAgent(
                    retriever=self.retriever,
                    llm=self.llm,
                    k=self._base_k,
                    context_budget_chars=self._base_context_chars,
                    system_prompt=(
                        "You are a technical summarizer. Improve faithfulness and coverage. "
                        "Explicitly ground claims with inline citations like [d4]. Be concise."
                    ),
                    use_mmr=True,
                    mmr_lambda=0.7,
                    embedding_fn=self.emb_fn,
                )
                retry = nudged.summarize(query, temperature=0.0, max_tokens=300)
                retry_summary = retry.summary

            hits2 = self.retriever.search(query, k=self._base_k)
            crit2 = self._analyst.critique(query=query, hits=hits2, summary=retry_summary)
            timeline.append({
                "step": "attempt_2",
                "used_docs": [d.doc.doc_id for d in hits2],
                "summary": retry_summary,
                "scores": {
                    "overall": crit2.scores.overall,
                    "faithfulness": crit2.scores.faithfulness,
                    "coverage": crit2.scores.coverage,
                    "clarity": crit2.scores.clarity,
                },
                "critic_latency_ms": crit2.llm.latency_ms,
                "critic_tokens": crit2.llm.total_tokens,
                "rewrite": crit2.suggested_rewrite,
                "justification": crit2.justification,
            })

            if crit2.scores.overall >= crit1.scores.overall:
                final_sum, final_crit = retry_summary, crit2

        # ---- compute + log reward ----
        reward = self._compute_reward(final_crit)
        self._log_reward(query, arm_name, reward, final_crit, attempts)

        return OrchestratorResult(
            query=query,
            attempts=attempts,
            final_summary=final_sum,
            final_scores={
                "overall": final_crit.scores.overall,
                "faithfulness": final_crit.scores.faithfulness,
                "coverage": final_crit.scores.coverage,
                "clarity": final_crit.scores.clarity,
                "reward": reward,
            },
            used_docs=[d.doc.doc_id for d in hits],
            timeline=timeline,
        )
