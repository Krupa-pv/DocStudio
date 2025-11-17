from __future__ import annotations
# simple ui to demo query -> retrieval -> summarize -> critic -> reward
# tries to stick to your existing syntax + objects

import os, time, json
from pathlib import Path
import streamlit as st

# load your stuff
from doc_studio.core.config import Settings
from doc_studio.core.logging import setup_logging
from doc_studio.core.llm import LLMClient
from doc_studio.retrieval import Document
from doc_studio.retrieval.hf_embedder import HFEmbedder
from doc_studio.retrieval.faiss_retriever import FaissRetriever
from doc_studio.orchestrator.orchestrate import Orchestrator, OrchestratorConfig

# optional: call FastAPI 
import requests

# quiet noisy HF warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# --- helpers ---
@st.cache_resource(show_spinner=False)
def _bootstrap() -> tuple[Settings, LLMClient, HFEmbedder, FaissRetriever]:
    cfg = Settings.load()
    setup_logging(getattr(cfg, "log_level", "INFO"))
    client = LLMClient(cfg)
    emb = HFEmbedder("BAAI/bge-small-en-v1.5")
    ret = FaissRetriever(emb)

    idx = Path("data/index.faiss"); meta = Path("data/index.meta.jsonl")
    if idx.exists() and meta.exists():
        ret.load(str(idx), str(meta))
    else:
        # tiny seed if no index yet
        seed = [
            Document("d1","RAG reduces hallucinations by grounding responses in retrieved documents."),
            Document("d2","MMR balances relevance and diversity to reduce redundancy in retrieval for RAG."),
            Document("d3","DPO optimizes policies directly from preferences without a reward model."),
            Document("d4","FlashAttention accelerates Transformer attention via IO-aware tiling on GPUs."),
            Document("d5","Hybrid retrieval combines BM25 with dense embeddings for better recall."),
        ]
        ret.add(seed)
    return cfg, client, emb, ret

def _make_orchestrator(ret: FaissRetriever, client: LLMClient, emb: HFEmbedder,
                       k:int, ctx_chars:int, mmr_lambda:float, threshold:float, use_rewrite:bool) -> Orchestrator:
    # stick to your class + reward fields + mmr path
    cfg = OrchestratorConfig(
        quality_threshold=threshold,
        max_attempts=2,
        use_rewrite_on_retry=use_rewrite,
        reward_alpha_tokens=0.02,
        reward_beta_latency=0.02,
    )
    # emb_fn enables MMR inside SummarizerAgent (your SummarizerAgent has use_mmr etc.)
    return Orchestrator(
        retriever=ret,
        llm=client,
        emb_fn=emb,
        cfg=cfg,
        reward_log_path="data/rewards.csv",
    )

# --- UI ---
st.set_page_config(page_title="docstudio RAG Demo", layout="wide")
st.title("docstudio — FAISS + MMR RAG (with Critic & Reward)")

cfg, client, emb, ret = _bootstrap()

with st.sidebar:
    st.subheader("Controls")
    backend_mode = st.radio("Backend", ["In-Process", "FastAPI /orchestrate"], index=0, help="use your local classes or call an API")
    k = st.slider("k (top docs)", 2, 8, 5)
    ctx_chars = st.slider("context budget (chars)", 600, 6000, 3000, step=200)
    mmr_lambda = st.slider("MMR λ (relevance vs diversity)", 0.0, 1.0, 0.6, step=0.05)
    qual_thresh = st.slider("quality threshold (retry if below)", 0.5, 0.95, 0.80, step=0.01)
    use_rewrite = st.checkbox("use analyst rewrite on retry", True)
    temp = st.slider("temperature (summarizer)", 0.0, 0.8, 0.0, step=0.1)
    max_tokens = st.slider("max_tokens (summarizer)", 100, 600, 300, step=50)
    st.divider()
    if st.button("Rebuild FAISS from ./data/corpus (run ingest script first)", type="secondary"):
        st.info("If you just ran scripts/ingest_faiss.py, restart the app (or trigger a change) to reload the saved index.")

query = st.text_input("Query", value="Explain how MMR reduces hallucinations in RAG. Provide citations.", label_visibility="visible")

colA, colB = st.columns([1, 2])
with colA:
    st.metric("Indexed docs", ret.size())
with colB:
    st.caption("Tip: run `python -m scripts.ingest_faiss --input_dir data/corpus` to persist a real corpus.")

run = st.button("Run Orchestrate", type="primary")

if run and query.strip():
    t0 = time.time()

    if backend_mode == "FastAPI /orchestrate":
        # call your API if running: uvicorn doc_studio.api.app:app --port 8000
        try:
            payload = {
                "query": query,
                "retry": True,
                "mmr_lambda_a": mmr_lambda,
                "mmr_lambda_b": mmr_lambda,
                "k_a": k,
                "k_b": max(k, k+1),
            }
            r = requests.post("http://127.0.0.1:8000/orchestrate", json=payload, timeout=60)
            r.raise_for_status()
            out = r.json()
            elapsed = (time.time() - t0) * 1000
        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()

        # display
        st.subheader("Answer")
        st.write(out["summary"])
        if out.get("rewrite"):
            with st.expander("Suggested rewrite"):
                st.write(out["rewrite"])

        st.subheader("Scores")
        s = out["scores"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overall", f'{s.get("overall",0):.2f}')
        c2.metric("Faithfulness", f'{s.get("faithfulness",0):.2f}')
        c3.metric("Coverage", f'{s.get("coverage",0):.2f}')
        c4.metric("Clarity", f'{s.get("clarity",0):.2f}')

        st.subheader("Meta")
        m1, m2, m3 = st.columns(3)
        m1.metric("Attempts", out.get("attempts", 1))
        m2.metric("Reward", f'{out.get("reward", 0):.3f}')
        usage = out.get("meta", {}).get("usage", {})
        m3.metric("Latency (ms)", f'{elapsed:.0f}')
        with st.expander("Used docs"):
            st.json(out.get("used_docs", []))
        with st.expander("Raw meta"):
            st.json(out.get("meta", {}))

    else:
        # In-Process path: use your existing Orchestrator with same CSV logging + fields
        orch = _make_orchestrator(ret, client, emb, k, ctx_chars, mmr_lambda, qual_thresh, use_rewrite)

        # we pass arm overrides via arm_params, same as your smoke
        arm_params = dict(k=k, context_budget_chars=ctx_chars, mmr_lambda=mmr_lambda, temperature=temp, max_tokens=max_tokens)
        res = orch.run(query, arm_name="ui", arm_params=arm_params)
        elapsed = (time.time() - t0) * 1000

        # --- display ---
        st.subheader("Answer")
        st.write(res.final_summary)

        st.subheader("Scores")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Overall", f'{res.final_scores["overall"]:.2f}')
        c2.metric("Faithfulness", f'{res.final_scores["faithfulness"]:.2f}')
        c3.metric("Coverage", f'{res.final_scores["coverage"]:.2f}')
        c4.metric("Clarity", f'{res.final_scores["clarity"]:.2f}')
        c5.metric("Reward", f'{res.final_scores["reward"]:.3f}')

        st.subheader("Meta")
        m1, m2 = st.columns(2)
        m1.metric("Attempts", res.attempts)
        m2.metric("Latency (ms)", f'{elapsed:.0f}')

        with st.expander("Used docs (ids)"):
            st.json(res.used_docs)

        st.subheader("Timeline")
        for step in res.timeline:
            with st.expander(f'{step["step"]} — overall={step["scores"]["overall"]:.3f}'):
                st.markdown("**Summary**")
                st.write(step["summary"])
                st.markdown("**Scores**")
                st.json(step["scores"])
                st.markdown("**Critic**")
                st.write(step.get("justification",""))
                meta_box = {
                    "critic_latency_ms": step.get("critic_latency_ms"),
                    "critic_tokens": step.get("critic_tokens"),
                }
                st.markdown("**Meta**")
                st.json(meta_box)

        st.caption("Reward log appended to data/rewards.csv")
