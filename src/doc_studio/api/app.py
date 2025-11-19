# src/doc_studio/api/app.py
"""
DocStudio Orchestrator API

This FastAPI application provides REST endpoints for:
1. Document upload and indexing (/upload_docs)
2. Orchestrated query processing (/orchestrate)
3. RL batch experiments (/rl/run)
4. Health checks (/health)

Integration points:
- Orchestrator: Uses orchestrate.py for query processing with retry logic
- Trace logging: All runs are logged via trace.py to data/traces.jsonl
- Policy: Supports epsilon-greedy bandit policy from policy.py
- Database: Logs queries to SQLite via db.py for UI history
- UI: Serves data to Streamlit UI app (ui/app.py) via HTTP
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from doc_studio.core.config import Settings
from doc_studio.core.logging import setup_logging
from doc_studio.core.llm import LLMClient
from doc_studio.retrieval import Document
from doc_studio.retrieval.faiss_retriever import FaissRetriever
from doc_studio.retrieval.hf_embedder import HFEmbedder
from doc_studio.orchestrator.orchestrate import Orchestrator, OrchestratorConfig
from doc_studio.orchestrator.policy import EpsilonGreedyPolicy, Arm
from doc_studio.db import insert_query_log


# --------- Pydantic models ---------

class OrchestrateRequest(BaseModel):
    query: str = Field(..., description="User query")
    arm_name: str = Field("default", description="Name of the retrieval/config arm")
    arm_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Override parameters for this arm (k, context_budget_chars, mmr_lambda, etc.)",
    )


class OrchestrateResponse(BaseModel):
    query: str
    final_summary: str
    final_scores: Dict[str, float]
    used_docs: List[Dict[str, Any]]
    timeline: List[Dict[str, Any]]


class UploadDoc(BaseModel):
    doc_id: str
    text: str
    meta: Optional[Dict[str, Any]] = None


class UploadDocsRequest(BaseModel):
    docs: List[UploadDoc]


class RLRunRequest(BaseModel):
    queries: List[str]
    epsilon: float = 0.1
    iterations: int = 200
    arms: Dict[str, Dict[str, Any]]  # {"A": {...}, "B": {...}, ...}


class RLRunResult(BaseModel):
    n_runs: int
    per_arm_stats: Dict[str, Dict[str, float]]
    csv_path: Optional[str] = None


# --------- Global state (backend singletons) ---------

app = FastAPI(title="DocStudio Orchestrator API")

_cfg: Optional[Settings] = None
_llm: Optional[LLMClient] = None
_emb: Optional[HFEmbedder] = None
_ret: Optional[FaissRetriever] = None
_orch: Optional[Orchestrator] = None


def _bootstrap() -> None:
    """Initialize settings, LLM, embedder, retriever, and orchestrator."""
    global _cfg, _llm, _emb, _ret, _orch
    if _orch is not None:
        return

    _cfg = Settings.load()
    setup_logging(getattr(_cfg, "log_level", "INFO"))

    _llm = LLMClient(_cfg)
    _emb = HFEmbedder("BAAI/bge-small-en-v1.5")
    _ret = FaissRetriever(_emb)

    idx_path = Path("data/index.faiss")
    meta_path = Path("data/index.meta.jsonl")
    if idx_path.exists() and meta_path.exists():
        _ret.load(str(idx_path), str(meta_path))
    else:
        # tiny seed corpus as fallback
        seed_docs = [
            Document("d1", "RAG reduces hallucinations by grounding answers in retrieved context."),
            Document("d2", "MMR balances relevance and diversity so the context is not redundant."),
            Document("d3", "DPO optimizes policies from preference data instead of a learned reward."),
        ]
        _ret.add(seed_docs)
        _ret.save(str(idx_path), str(meta_path))

    _orch = Orchestrator(
        retriever=_ret,
        llm=_llm,
        emb_fn=_emb,
        cfg=OrchestratorConfig(),
        reward_log_path="data/rewards.csv",
    )


@app.on_event("startup")
async def on_startup() -> None:
    _bootstrap()


# --------- Health ---------

@app.get("/health")
async def health() -> Dict[str, Any]:
    _bootstrap()
    return {
        "status": "ok",
        "docs_indexed": _ret.size() if _ret else 0,
    }


# --------- Document upload ---------

@app.post("/upload_docs")
async def upload_docs(req: UploadDocsRequest) -> Dict[str, Any]:
    _bootstrap()
    assert _ret is not None

    docs = [
        Document(d.doc_id, d.text, d.meta)
        for d in req.docs
    ]
    _ret.add(docs)

    idx_path = Path("data/index.faiss")
    meta_path = Path("data/index.meta.jsonl")
    _ret.save(str(idx_path), str(meta_path))

    return {
        "status": "ok",
        "added": len(docs),
        "total_docs": _ret.size(),
    }


# --------- Single orchestrator run ---------

@app.post("/orchestrate", response_model=OrchestrateResponse)
async def orchestrate(req: OrchestrateRequest) -> OrchestrateResponse:
    _bootstrap()
    assert _orch is not None

    # Validate query
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    res = _orch.run(
        query=req.query.strip(),
        arm_name=req.arm_name,
        arm_params=req.arm_params,
    )

    # Normalize used_docs into plain dicts for the API
    used_docs_payload: List[Dict[str, Any]] = []
    for d in res.used_docs:
        if isinstance(d, dict):
            used_docs_payload.append(d)
        else:
            # fallback in case OrchestratorResult.used_docs is a list of ids
            used_docs_payload.append({"doc_id": str(d)})

    # Log into queries table (db.py insert_query_log already ensures table)
    try:
        insert_query_log(
            query=res.query,
            answer=res.final_summary,
            scores=res.final_scores,
            arm_name=req.arm_name,
        )
    except Exception as e:
        # Don't crash the API because of logging issues
        import logging
        logging.warning(f"Failed to log query to database: {e}")

    return OrchestrateResponse(
        query=res.query,
        final_summary=res.final_summary,
        final_scores=res.final_scores,
        used_docs=used_docs_payload,
        timeline=res.timeline,
    )


# --------- RL batch run endpoint ---------

@app.post("/rl/run", response_model=RLRunResult)
async def rl_run(req: RLRunRequest) -> RLRunResult:
    """
    Run an ε-greedy bandit over the orchestrator for a set of queries and arms.
    Useful if you want to trigger RL experiments from the UI / scripts.
    """
    _bootstrap()
    assert _orch is not None

    import random, time, csv
    import numpy as np

    random.seed(7)
    np.random.seed(7)

    # Build policy
    policy = EpsilonGreedyPolicy(
        epsilon=req.epsilon,
        reward_log_path="data/rl_bandit_rewards.csv",
    )

    for name, params in req.arms.items():
        policy.add_arm(Arm(name=name, params=params))

    records: List[Dict[str, Any]] = []
    for t in range(req.iterations):
        arm = policy.choose()
        q = random.choice(req.queries)

        t0 = time.time()
        res = _orch.run(q, arm_name=arm.name, arm_params=arm.params)
        elapsed_ms = (time.time() - t0) * 1000.0

        s = res.final_scores
        reward = float(s.get("reward", 0.0))

        policy.update(arm.name, reward)

        records.append(
            {
                "t": t,
                "query": q,
                "arm": arm.name,
                "reward": reward,
                "overall": s.get("overall", 0.0),
                "faithfulness": s.get("faithfulness", 0.0),
                "coverage": s.get("coverage", 0.0),
                "clarity": s.get("clarity", 0.0),
                "latency_ms": elapsed_ms,
            }
        )

    # Optionally dump to CSV for offline plots
    out_csv = "data/rl_runs_api.csv"
    Path("data").mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    # Summarize per-arm
    from collections import defaultdict

    per_arm: Dict[str, Dict[str, float]] = defaultdict(lambda: {"mean_reward": 0.0, "n": 0})
    for r in records:
        a = r["arm"]
        per_arm[a]["mean_reward"] += r["reward"]
        per_arm[a]["n"] += 1

    for a, stats in per_arm.items():
        if stats["n"] > 0:
            stats["mean_reward"] /= stats["n"]

    return RLRunResult(
        n_runs=len(records),
        per_arm_stats=per_arm,
        csv_path=out_csv,
    )
