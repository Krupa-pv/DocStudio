from __future__ import annotations
import csv, os, time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from doc_studio.core.trace import TraceLogger
from doc_studio.core.llm import LLMClient
from doc_studio.core.types import ChatMessage, Role
from doc_studio.retrieval import Retriever, Document, ScoredDocument
from doc_studio.agents.summarizer import SummarizerAgent, SummaryResult  # <-- MMR-aware version
from doc_studio.agents.analyst import AnalystAgent, CritiqueResult
from doc_studio.tracking.run_logger import log_run
from doc_studio.db import insert_query_log


@dataclass
class OrchestratorConfig:
    quality_threshold: float = 0.75
    max_attempts: int = 2
    use_rewrite_on_retry: bool = True

    # efficiency-aware reward weights
    reward_alpha_tokens: float = 0.02   # penalty per 1k tokens
    reward_beta_latency: float = 0.02   # penalty per second

    # defaults for summarizer
    base_k: int = 5
    base_context_chars: int = 3000
    base_system_prompt: str = (
        "You are a technical summarizer. Write concise, factual summaries with inline citations "
        "using the provided document IDs in square brackets, e.g., [d4][d2]. Do not invent citations."
    )


@dataclass
class OrchestratorAttempt:
    query: str
    summary: str
    critique: CritiqueResult
    used_docs: List[ScoredDocument]
    arm_name: str
    arm_params: Dict[str, Any]
    elapsed_ms: float


@dataclass
class OrchestratorResult:
    query: str
    attempts: int
    final_summary: str
    final_scores: Dict[str, float]
    used_docs: List[str]
    timeline: List[Dict[str, Any]]
    total_tokens: int
    latency_ms: float



class Orchestrator:
    """
    Orchestrates:
      query -> retrieval (SummarizerAgent) -> summary -> critique (AnalystAgent) -> reward
    And logs everything to:
      - CSV reward log
      - SQLite runs table (via run_logger)
      - queries table (for history UI)
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        emb_fn,
        *,
        cfg: Optional[OrchestratorConfig] = None,
        reward_log_path: str = "data/rewards.csv",
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.emb_fn = emb_fn
        self.cfg = cfg or OrchestratorConfig()
        self.reward_log_path = reward_log_path
        self.tracer = TraceLogger(out_path="data/traces.jsonl")
        self._base_k = self.cfg.base_k
        self._base_context_chars = self.cfg.base_context_chars
        self._base_system_prompt = self.cfg.base_system_prompt

        # default summarizer (MMR-enabled)
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
            with open(self.reward_log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "ts_utc",
                        "query",
                        "arm",
                        "overall",
                        "faithfulness",
                        "coverage",
                        "clarity",
                        "reward",
                        "tokens_prompt",
                        "tokens_completion",
                        "latency_ms",
                        "attempts",
                    ],
                )
                writer.writeheader()

    def _compute_reward(self, crit: CritiqueResult) -> float:
        """
        Reward = overall - α * (tokens / 1k) - β * (latency / 1s)

        Where:
          - overall is the critic's blended quality score (0..1)
          - tokens / latency come from the LLM ChatResult
        """
        toks_pen = self.cfg.reward_alpha_tokens * (crit.llm.total_tokens / 1000.0)
        lat_pen = self.cfg.reward_beta_latency * (crit.llm.latency_ms / 1000.0)
        return max(0.0, crit.scores.overall - toks_pen - lat_pen)

    def _log_reward(
        self,
        query: str,
        arm_name: str,
        reward: float,
        crit: CritiqueResult,
        attempts: int,
    ) -> None:
        """Low-level CSV reward log for quick inspection."""
        with open(self.reward_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "ts_utc",
                    "query",
                    "arm",
                    "overall",
                    "faithfulness",
                    "coverage",
                    "clarity",
                    "reward",
                    "tokens_prompt",
                    "tokens_completion",
                    "latency_ms",
                    "attempts",
                ],
            )
            writer.writerow(
                {
                    "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "query": query,
                    "arm": arm_name,
                    "overall": crit.scores.overall,
                    "faithfulness": crit.scores.faithfulness,
                    "coverage": crit.scores.coverage,
                    "clarity": crit.scores.clarity,
                    "reward": reward,
                    "tokens_prompt": crit.llm.prompt_tokens,
                    "tokens_completion": crit.llm.completion_tokens,
                    "latency_ms": crit.llm.latency_ms,
                    "attempts": attempts,
                }
            )

    def _configure_summarizer_for_arm(self, arm_params: Dict[str, Any]) -> None:
        """
        Update summarizer config according to the chosen arm.
        """
        self._summarizer.k = arm_params.get("k", self._base_k)
        self._summarizer.context_budget_chars = arm_params.get(
            "context_budget_chars", self._base_context_chars
        )
        self._summarizer.system_prompt = arm_params.get(
            "system_prompt", self._base_system_prompt
        )
        self._summarizer.use_mmr = arm_params.get("use_mmr", True)
        self._summarizer.mmr_lambda = arm_params.get("mmr_lambda", 0.6)
        self._summarizer.embedding_fn = self.emb_fn

    def run(
        self,
        query: str,
        *,
        arm_name: str = "default",
        arm_params: Optional[Dict[str, Any]] = None,
    ) -> OrchestratorResult:
        """
        Run a full loop:
            1) Summarize under a given arm config
            2) Critique
            3) Optional retry with rewrite if below quality threshold
            4) Compute reward (quality – efficiency penalties)
            5) Log to CSV, SQLite, and MLflow
        """
        arm_params = arm_params or {}
        attempts: List[OrchestratorAttempt] = []
        timeline: List[Dict[str, Any]] = []

        trace_run = self.tracer.start(query)

        # configure summarizer for this arm
        self._configure_summarizer_for_arm(arm_params)
        trace_run.record(
            phase="arm_config",
            arm_name=arm_name,
            arm_params=arm_params,
            k=self._summarizer.k,
            ctx_chars=self._summarizer.context_budget_chars,
        )

        start = time.time()

        # ---- First attempt ----
        sum1 = self._summarizer.summarize(
            query,
            temperature=float(arm_params.get("temperature", 0.2)),
            max_tokens=int(arm_params.get("max_tokens", 400)),
        )
        trace_run.record(
            phase="summarize_initial",
            arm_name=arm_name,
            summary=sum1.summary,
            used_docs=[d_id for (d_id, _score) in sum1.used_docs],
        )

        hits = [
            ScoredDocument(doc=Document(doc_id=d_id, text="", meta={}), score=score)
            for (d_id, score) in sum1.used_docs
        ]

        crit1 = self._analyst.critique(
            query,
            hits=self.retriever.search(query, k=self._summarizer.k),
            summary=sum1.summary,
            temperature=0.0,
            max_tokens=400,
            context_override=sum1.context_str,
        )

        trace_run.record(
            phase="critique_initial",
            arm_name=arm_name,
            scores=crit1.scores.__dict__,
            tokens=crit1.llm.total_tokens,
            latency_ms=crit1.llm.latency_ms,
        )

        attempts.append(
            OrchestratorAttempt(
                query=query,
                summary=sum1.summary,
                critique=crit1,
                used_docs=hits,
                arm_name=arm_name,
                arm_params=arm_params,
                elapsed_ms=(time.time() - start) * 1000.0,
            )
        )
        timeline.append(
            {
                "step": "initial",
                "summary": sum1.summary,
                "scores": crit1.scores.__dict__,
                "latency_ms": crit1.llm.latency_ms,
                "tokens": crit1.llm.total_tokens,
            }
        )

        final_summary = sum1.summary
        final_crit = crit1
        attempts_count = 1

        # ---- Optional retry with rewrite ----
        if (
            self.cfg.use_rewrite_on_retry
            and crit1.scores.overall < self.cfg.quality_threshold
            and self.cfg.max_attempts > 1
        ):
            if crit1.suggested_rewrite:
                retry_prompt = crit1.suggested_rewrite
            else:
                retry_prompt = sum1.summary

            sum2 = self.llm.chat(
                messages=[
                    ChatMessage(role=Role.SYSTEM, content=self._base_system_prompt),
                    ChatMessage(
                        role=Role.USER,
                        content=f"Rewrite the following answer to improve faithfulness and clarity.\n\n{retry_prompt}",
                    ),
                ],
                temperature=float(arm_params.get("temperature", 0.2)),
                max_tokens=int(arm_params.get("max_tokens", 400)),
            )

            trace_run.record(
                phase="rewrite_retry",
                arm_name=arm_name,
                rewritten=retry_prompt,
                answer=sum2.text,
            )

            crit2 = self._analyst.critique(
                query,
                hits=self.retriever.search(query, k=self._summarizer.k),
                summary=sum2.text,
                temperature=0.0,
                max_tokens=400,
                context_override=sum1.context_str,
            )

            trace_run.record(
                phase="critique_retry",
                arm_name=arm_name,
                scores=crit2.scores.__dict__,
                tokens=crit2.llm.total_tokens,
                latency_ms=crit2.llm.latency_ms,
            )

            attempts_count = 2
            final_summary = sum2.text
            final_crit = crit2
            timeline.append(
                {
                    "step": "retry",
                    "summary": sum2.text,
                    "scores": crit2.scores.__dict__,
                    "latency_ms": crit2.llm.latency_ms,
                    "tokens": crit2.llm.total_tokens,
                }
            )

        elapsed_ms = (time.time() - start) * 1000.0

        # ---- compute + log reward ----
        reward = self._compute_reward(final_crit)
        self._log_reward(query, arm_name, reward, final_crit, attempts_count)

        trace_run.record(
            phase="final",
            arm_name=arm_name,
            reward=reward,
            overall=final_crit.scores.overall,
            faithfulness=final_crit.scores.faithfulness,
            coverage=final_crit.scores.coverage,
            clarity=final_crit.scores.clarity,
            latency_ms=elapsed_ms,
            attempts=attempts_count,
        )
        trace_run.finish(reward=reward)
        self.tracer.save(trace_run)

        p_k = arm_params.get("k", self._base_k)
        p_mmr = arm_params.get("mmr_lambda", getattr(self._summarizer, "mmr_lambda", 0.6))
        params_for_log = {
            "arm": arm_name,
            "k": p_k,
            "mmr_lambda": p_mmr,
            "retriever": "faiss_flatip",
            "llm": "azure_gpt4o",
        }
        scores_for_log = {
            "overall": float(final_crit.scores.overall),
            "faithfulness": float(final_crit.scores.faithfulness),
            "coverage": float(final_crit.scores.coverage),
            "clarity": float(final_crit.scores.clarity),
            "reward": float(reward),
        }

        log_run(
            query=query,
            params=params_for_log,
            scores=scores_for_log,
            tokens_prompt=int(final_crit.llm.prompt_tokens),
            tokens_completion=int(final_crit.llm.completion_tokens),
            latency_ms=int(elapsed_ms),
        )

        try:
            insert_query_log(
                query=query,
                answer=final_summary,
                scores={
                    "overall": float(final_crit.scores.overall),
                    "faithfulness": float(final_crit.scores.faithfulness),
                    "coverage": float(final_crit.scores.coverage),
                    "clarity": float(final_crit.scores.clarity),
                    "reward": float(reward),
                },
                arm_name=arm_name,
            )
        except Exception as e:
            import logging
            logging.warning(f"Failed to log query to database: {e}")

        
        return OrchestratorResult(
            query=query,
            attempts=attempts_count,          # not the list; just the number of attempts
            final_summary=final_summary,
            final_scores={
                "overall": final_crit.scores.overall,
                "faithfulness": final_crit.scores.faithfulness,
                "coverage": final_crit.scores.coverage,
                "clarity": final_crit.scores.clarity,
                "reward": reward,
            },
            used_docs=[d.doc.doc_id for d in hits],
            timeline=timeline,
            total_tokens=int(final_crit.llm.total_tokens),
            latency_ms=elapsed_ms,
        )




