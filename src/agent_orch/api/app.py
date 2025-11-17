from __future__ import annotations

from typing import Optional, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel, Field

from doc_studio.core.config import Settings
from doc_studio.core.logging import setup_logging
from doc_studio.core.llm import LLMClient
from doc_studio.retrieval import Document, TfidfRetriever
from doc_studio.orchestrator.orchestrate import Orchestrator, OrchestratorConfig
from doc_studio.orchestrator.policy import EpsilonGreedyPolicy, Arm
from pathlib import Path

from src.doc_studio.retrieval.faiss_retriever import FaissRetriever
from src.doc_studio.retrieval.hf_embedder import HFEmbedder

emb = HFEmbedder("BAAI/bge-small-en-v1.5")
ret = FaissRetriever(emb)

IDX=Path("data/index.faiss"); META=Path("data/index.meta.jsonl")
if IDX.exists() and META.exists():
    # load persisted
    ret.load(str(IDX), str(META))
else:
    # fallback tiny seed
    _SEED = [
        Document("d1","rag reduces hallucinations by grounding answers in retrieved context."),
        Document("d2","mmr balances relevance and diversity so the context isn’t redundant for rag."),
        Document("d3","dpo optimizes from preference pairs and skips a reward model."),
        Document("d4","flashattention speeds up attention with io-aware tiling on gpus."),
        Document("d5","hybrid retrieval mixes bm25 keywords with dense embeddings to boost recall."),
    ]
    ret.add(_SEED)

# ---------- Request/Response models ----------
class OrchestrateRequest(BaseModel):
    query: str = Field(..., min_length=3)
    use_bandit: bool = False
    epsilon: float = 0.2
    arm_override: Optional[str] = None  # if you want to force a specific arm
    params: Dict[str, Any] = {}         # optional per-run knobs (k, temperature, etc.)


class OrchestrateResponse(BaseModel):
    query: str
    attempts: int
    final_summary: str
    final_scores: Dict[str, float]
    used_docs: list[str]
    timeline: list[dict]


# ---------- App factory ----------
def create_app() -> FastAPI:
    cfg = Settings.load()
    setup_logging(cfg.log_level)

    # Build shared components once (simple in-memory retriever demo corpus)
    retriever = TfidfRetriever()
    retriever.add([
        Document("d1", "LoRA and QLoRA are lightweight fine-tuning methods for LLMs."),
        Document("d2", "Reinforcement learning from human feedback (RLHF) aligns models with human preferences."),
        Document("d3", "FlashAttention accelerates Transformer attention via IO-aware tiling on GPUs."),
        Document("d4", "Maximal Marginal Relevance (MMR) balances relevance and diversity for retrieval in RAG systems, which can reduce hallucinations by providing diverse evidence."),
        Document("d5", "DPO optimizes policies from preferences without an explicit reward model."),
    ])

    client = LLMClient(cfg)
    orch = Orchestrator(
        retriever=retriever,
        llm=client,
        cfg=OrchestratorConfig(quality_threshold=0.80, max_attempts=2, use_rewrite_on_retry=True),
        reward_log_path="data/rewards.csv",
    )

    # Define a couple of arms for bandit mode
    arms = [
        Arm("k3_temp0_rewrite", {"k": 3, "temperature": 0.0, "max_tokens": 280, "context_budget_chars": 1400, "use_rewrite_on_retry": True}),
        Arm("k5_temp0_rewrite", {"k": 5, "temperature": 0.0, "max_tokens": 300, "context_budget_chars": 3000, "use_rewrite_on_retry": True}),
        Arm("k5_temp03_norewrite", {"k": 5, "temperature": 0.3, "max_tokens": 320, "context_budget_chars": 3000, "use_rewrite_on_retry": False}),
    ]
    policy = EpsilonGreedyPolicy(arms=arms, epsilon=0.25, reward_log_path="data/rewards.csv")

    app = FastAPI(title="docstudio API", version="0.1.0")

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.post("/orchestrate", response_model=OrchestrateResponse)
    def orchestrate(req: OrchestrateRequest):
        # choose arm
        if req.use_bandit and not req.arm_override:
            arm = policy.choose()
            arm_name, arm_params = arm.name, {**arm.params, **req.params}
        else:
            # fallback to an explicit or default arm
            arm_name = req.arm_override or "default"
            arm_params = req.params

        result = orch.run(query=req.query, arm_name=arm_name, arm_params=arm_params)

        return OrchestrateResponse(
            query=result.query,
            attempts=result.attempts,
            final_summary=result.final_summary,
            final_scores=result.final_scores,
            used_docs=result.used_docs,
            timeline=result.timeline,
        )

    return app


app = create_app()
