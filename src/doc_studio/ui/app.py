from __future__ import annotations
import altair as alt
import random
from typing import Any, Tuple
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from pandas.errors import DatabaseError
import os, time, json
from pathlib import Path
import streamlit as st
import sqlite3

# load your stuff
from doc_studio.core.config import Settings
from doc_studio.core.logging import setup_logging
from doc_studio.core.llm import LLMClient
from doc_studio.db import DB_PATH
from doc_studio.retrieval import Document
from doc_studio.retrieval.hf_embedder import HFEmbedder
from doc_studio.retrieval.faiss_retriever import FaissRetriever
from doc_studio.orchestrator.orchestrate import Orchestrator, OrchestratorConfig
from doc_studio.orchestrator.policy import EpsilonGreedyPolicy, Arm
# optional: call FastAPI
import requests

ALPHA_PER_KTOK = 0.02   # penalty per 1k tokens
BETA_PER_SEC    = 0.02  # penalty per second
API_URL = os.getenv("STREAMLIT_API_URL", "http://localhost:8000")

def compute_reward(overall: float, tokens: int, latency_ms: float) -> float:
    return float(overall) - ALPHA_PER_KTOK * (tokens / 1000.0) - BETA_PER_SEC * (latency_ms / 1000.0)


# quiet noisy HF warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@st.cache_resource
def _bootstrap() -> Tuple[Settings, LLMClient, HFEmbedder, FaissRetriever]:
    """
    Initialize and cache the core components.
    Returns: (cfg, llm_client, embedder, retriever)
    """
    cfg = Settings.load()
    setup_logging(cfg.log_level)

    client = LLMClient(cfg)
    emb = HFEmbedder("BAAI/bge-small-en-v1.5")
    ret = FaissRetriever(emb)

    # Load index if exists
    idx_path = Path("data/index.faiss")
    meta_path = Path("data/index.meta.jsonl")
    if idx_path.exists() and meta_path.exists():
        ret.load(str(idx_path), str(meta_path))
    else:
        # seed corpus fallback
        seed_docs = [
            Document("d1", "LoRA and QLoRA are lightweight fine-tuning methods for LLMs."),
            Document("d2", "Reinforcement learning from human feedback (RLHF) aligns models with human preferences."),
            Document("d3", "FlashAttention accelerates Transformer attention via IO-aware tiling on GPUs."),
            Document("d4", "Maximal Marginal Relevance (MMR) balances relevance and diversity for retrieval in RAG systems."),
            Document("d5", "DPO optimizes policies from preferences without an explicit reward model."),
        ]
        ret.add(seed_docs)

    return cfg, client, emb, ret




def rl_learning_dashboard(log_path: str = "data/rl_bandit_rewards.csv") -> None:
    """
    Visualize bandit policy learning: shows how the RL algorithm discovers optimal configurations

    This displays the running mean reward per arm over time, demonstrating exploration-exploitation
    tradeoffs as the policy learns which configuration achieves best quality-efficiency balance.

    Expected columns: ts_utc, arm, reward, count, mean_reward, epsilon
    Data source: EpsilonGreedyPolicy or UCBPolicy CSV logs
    """

    path = Path(log_path)
    if not path.exists():
        st.info(f"No bandit reward log found at {log_path}. Run the RL Tuning Lab first.")
        return

    df = pd.read_csv(path)
    if df.empty:
        st.info("Bandit log is empty. Run the RL Tuning Lab for a while first.")
        return

    # Add episode index
    if "episode" not in df.columns:
        df["episode"] = range(1, len(df) + 1)

    # Show key insight
    st.info("📈 Learning Curve: Watch how the running mean reward (orange line) increases as the policy learns which configurations perform best")

    # Running mean reward (this is what the policy actually uses)
    st.markdown("**Running Mean Reward (Policy Learning)**")
    chart_mean = (
        alt.Chart(df)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("episode:Q", title="Episode"),
            y=alt.Y("mean_reward:Q", title="Running Mean Reward", scale=alt.Scale(zero=False)),
            color=alt.Color("arm:N", title="Arm"),
            tooltip=["episode", "arm", "mean_reward", "count"]
        )
        .interactive()
    )
    st.altair_chart(chart_mean, use_container_width=True)

    # Individual reward observations (noisy)
    st.markdown("**Individual Rewards per Episode**")
    chart_reward = (
        alt.Chart(df)
        .mark_circle(opacity=0.6, size=60)
        .encode(
            x=alt.X("episode:Q", title="Episode"),
            y=alt.Y("reward:Q", title="Reward (individual)"),
            color=alt.Color("arm:N", title="Arm"),
            tooltip=["episode", "arm", "reward", "count"]
        )
        .interactive()
    )
    st.altair_chart(chart_reward, use_container_width=True)

    # Per-arm mean reward from log itself
    st.markdown("**Mean Reward per Arm (from log)**")
    arm_means = df.groupby("arm")["reward"].mean().reset_index()
    chart_means = (
        alt.Chart(arm_means)
        .mark_bar()
        .encode(
            x=alt.X("arm:N", title="Arm"),
            y=alt.Y("reward:Q", title="Mean reward"),
            color="arm:N",
            tooltip=["arm", "reward"],
        )
    )
    st.altair_chart(chart_means, use_container_width=True)

    # Arm selection counts
    st.markdown("**Arm Selection Counts**")
    counts = df["arm"].value_counts().reset_index()
    counts.columns = ["arm", "count"]
    chart_counts = (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("arm:N", title="Arm"),
            y=alt.Y("count:Q", title="Selections"),
            color="arm:N",
            tooltip=["arm", "count"],
        )
    )
    st.altair_chart(chart_counts, use_container_width=True)



def history_tab() -> None:
    st.header("History & Analytics")

    if not DB_PATH.exists():
        st.info("No history yet. Run some queries in the Studio or RL Tuning Lab first.")
        return

    conn = sqlite3.connect(DB_PATH)
    try:
        try:
            df = pd.read_sql_query(
                """
                SELECT created_at, query, arm_name, overall, reward
                FROM queries
                ORDER BY id DESC
                LIMIT 200
                """,
                conn,
            )
        except (sqlite3.OperationalError, DatabaseError):
            st.info(
                "The `queries` table does not exist yet. "
                "Run the orchestrator at least once so queries can be logged."
            )
            return
    finally:
        conn.close()

    if df.empty:
        st.info("No queries logged yet. Run the orchestrator to populate history.")
        return

    st.subheader("Recent queries")
    st.dataframe(df)

    # Light-weight reward-over-time chart (for *real* queries)
    st.subheader("Reward over time (logged queries)")
    df_time = df.sort_values("created_at").set_index("created_at")
    st.line_chart(df_time[["reward"]])

    # Simple per-arm summary from actual usage (if any arm_name)
    if "arm_name" in df.columns and df["arm_name"].notnull().any():
        st.subheader("Mean reward per arm (from query logs)")
        arm_stats = (
            df.groupby("arm_name")["reward"]
            .mean()
            .sort_values(ascending=False)
        )
        st.bar_chart(arm_stats)
def rl_eval_section(csv_path: str = "data/rl_runs_api.csv") -> None:
    """
    Display detailed per-query execution logs from RL experiments

    This shows individual query results including quality scores, tokens, latency,
    and which arm (configuration) was selected for each query. Complements the
    bandit policy learning curves by showing execution details.

    Expected columns: t, query, arm, overall, reward, latency_ms, tokens_*
    Data source: API endpoint /rl/run logs
    """

    path = Path(csv_path)
    if not path.exists():
        st.info(
            f"No RL runs CSV found at {csv_path}. "
            "Run an experiment in the RL Tuning Lab first."
        )
        return

    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.warning(f"Could not read RL CSV at {csv_path}: {e}")
        return

    if df.empty:
        st.info("RL runs CSV is empty. Run the RL Tuning Lab to generate data.")
        return

    # Reward vs iteration
    if "t" in df.columns and "reward" in df.columns:
        st.markdown("**Reward vs iteration**")
        st.line_chart(df.set_index("t")[["reward"]])

    # Optional: overall vs tokens
    if "overall" in df.columns and "latency_ms" in df.columns:
        st.markdown("**Quality vs latency scatter**")
        st.scatter_chart(df[["latency_ms", "overall"]])

    # Arm selection & per-arm stats
    if "arm" in df.columns:
        st.markdown("**Arm selection share**")
        freq = df["arm"].value_counts(normalize=True)
        st.bar_chart(freq)

        st.markdown("**Per-arm summary**")
        by_arm = (
            df.groupby("arm")
            .agg(
                mean_overall=("overall", "mean"),
                mean_reward=("reward", "mean"),
                mean_latency=("latency_ms", "mean"),
            )
            .reset_index()
        )
        st.dataframe(by_arm)
# ---- UI ----

st.set_page_config(page_title="DocStudio RL Orchestrator", layout="wide")
st.title("DocStudio — RL-Tuned RAG Orchestrator")

# Backend health check in sidebar
with st.sidebar:
    st.subheader("Backend Status")
    try:
        health_resp = requests.get(f"{API_URL}/health", timeout=5)
        if health_resp.ok:
            health_data = health_resp.json()
            st.success("Connected")
            st.metric("Documents Indexed", health_data.get("docs_indexed", 0))
        else:
            st.error(f"Backend error: {health_resp.status_code}")
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to {API_URL}")
        st.caption("Make sure the API server is running")
    except Exception as e:
        st.warning(f"Health check failed: {e}")

    st.markdown("---")
    st.caption(f"API URL: `{API_URL}`")
    st.caption("Set via `STREAMLIT_API_URL` env var")

tabs = st.tabs(["Upload Docs", "Studio", "RL Tuning Lab", "Evaluation", "History"])

with tabs[0]:
    st.title("Upload Documents for Analysis")
    uploaded_files = st.file_uploader("Upload .txt or .md", type=["txt", "md"], accept_multiple_files=True)

    if uploaded_files and st.button("Send to backend index"):
        docs_payload = []
        for f in uploaded_files:
            text = f.read().decode("utf-8", errors="ignore")
            docs_payload.append({
                "doc_id": f.name,
                "text": text,
                "meta": {"filename": f.name},
            })

        resp = requests.post(
            f"{API_URL}/upload_docs",
            json={"docs": docs_payload},
            timeout=60,
        )
        if resp.ok:
            data = resp.json()
            st.success(f"Indexed {data.get('added', 0)} docs (total {data.get('total_docs', '?')}).")
        else:
            st.error(f"Upload failed: {resp.status_code} {resp.text}")


with tabs[1]:
    st.title("DocStudio — Orchestrator (via API)")

    query = st.text_area("Question", placeholder="Enter your question here...")
    arm_name = st.text_input("Arm name", value="default")

    k = st.slider("k (top docs)", 2, 12, 5)
    ctx_chars = st.slider("Context chars", 500, 6000, 3000, 250)
    mmr_lambda = st.slider("MMR λ", 0.0, 1.0, 0.6, 0.05)
    temp = st.slider("LLM temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.slider("Max answer tokens", 100, 800, 400, 50)

    if st.button("Run Orchestrator", type="primary"):
        if not query or not query.strip():
            st.error("Please enter a question before running the orchestrator.")
        else:
            payload = {
                "query": query.strip(),
                "arm_name": arm_name,
                "arm_params": {
                    "k": k,
                    "context_budget_chars": ctx_chars,
                    "mmr_lambda": mmr_lambda,
                    "temperature": temp,
                    "max_tokens": max_tokens,
                },
            }
            try:
                with st.spinner("Calling backend /orchestrate..."):
                    resp = requests.post(f"{API_URL}/orchestrate", json=payload, timeout=120)

                if not resp.ok:
                    st.error(f"Backend error: {resp.status_code} {resp.text}")
                else:
                    data = resp.json()
                    st.subheader("Final Answer")
                    st.write(data["final_summary"])

                    st.subheader("Scores")
                    s = data["final_scores"]
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("Overall", f"{s.get('overall', 0):.2f}")
                    c2.metric("Faithfulness", f"{s.get('faithfulness', 0):.2f}")
                    c3.metric("Coverage", f"{s.get('coverage', 0):.2f}")
                    c4.metric("Clarity", f"{s.get('clarity', 0):.2f}")
                    c5.metric("Reward", f"{s.get('reward', 0):.2f}")

                    st.subheader("Timeline")
                    st.json(data["timeline"])

                    st.subheader("Used docs")
                    st.json(data["used_docs"])
            except requests.exceptions.Timeout:
                st.error("Request timed out. The backend may be overloaded or the query is too complex.")
            except requests.exceptions.ConnectionError:
                st.error(f"Could not connect to backend at {API_URL}. Make sure the API server is running.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")


with tabs[2]:
    st.header("RL Tuner — ε-greedy planner")

    # ---- Controls
    col = st.columns(4)
    with col[0]:
        epsilon = st.slider("ε (exploration)", 0.0, 0.5, 0.1, 0.01)
    with col[1]:
        iters = st.number_input("Iterations", min_value=50, max_value=5000, value=600, step=50)
    with col[2]:
        seed = st.number_input("Seed", min_value=0, max_value=2_000_000, value=7, step=1)
    with col[3]:
        use_eval_file = st.checkbox("Use eval.jsonl", value=True)

    random.seed(seed)
    np.random.seed(seed)

    cfg, client, emb, ret = _bootstrap()
    orch = Orchestrator(retriever=ret, llm=client, emb_fn=emb)

    if not use_eval_file:
        qtext = st.text_area(
            "Queries (one per line)",
            value=(
                "Explain how MMR reduces hallucinations in RAG.\n"
                "RLHF vs DPO for alignment.\n"
                "How does FlashAttention speed up attention?\n"
                "Embeddings vs TF-IDF for search.\n"
            ),
        )
        queries = [q.strip() for q in qtext.splitlines() if q.strip()]
    else:
        evalp = Path("data/eval.jsonl")
        if not evalp.exists():
            st.warning("data/eval.jsonl not found; falling back to defaults from the text box.")
            queries = [
                "Explain how MMR reduces hallucinations in RAG.",
                "RLHF vs DPO",
                "FlashAttention on GPUs",
            ]
        else:
            queries = [json.loads(L)["query"] for L in evalp.read_text().splitlines() if L.strip()]

    st.subheader("Arms (configs planner can choose)")
    st.caption("You can tune these. Keep them small to see learning clearly.")
    colA, colB = st.columns(2)

    with colA:
        k_a = st.slider("[A] k", 2, 8, 3)
        temp_a = st.slider("[A] temp", 0.0, 0.7, 0.0, 0.1)
        ctx_a = st.slider("[A] ctx chars", 600, 6000, 1400, 200)
        mmr_a = st.slider("[A] MMR λ", 0.0, 1.0, 0.6, 0.05)
    with colB:
        k_b = st.slider("[B] k", 2, 8, 5)
        temp_b = st.slider("[B] temp", 0.0, 0.7, 0.0, 0.1)
        ctx_b = st.slider("[B] ctx chars", 600, 6000, 3000, 200)
        mmr_b = st.slider("[B] MMR λ", 0.0, 1.0, 0.6, 0.05)

    add_arm_c = st.checkbox("Add Arm C", value=False)
    if add_arm_c:
        k_c = st.slider("[C] k", 2, 8, 4)
        temp_c = st.slider("[C] temp", 0.0, 0.7, 0.0, 0.1)
        ctx_c = st.slider("[C] ctx chars", 600, 6000, 2000, 200)
        mmr_c = st.slider("[C] MMR λ", 0.0, 1.0, 0.6, 0.05)

    if st.button("Run RL Tuning", type="primary"):
        policy = EpsilonGreedyPolicy(
            epsilon=epsilon,
            reward_log_path="data/rl_bandit_rewards.csv",
        )

        policy.add_arm(
            Arm(
                name="A",
                params={
                    "k": k_a,
                    "context_budget_chars": ctx_a,
                    "mmr_lambda": mmr_a,
                    "temperature": temp_a,
                    "max_tokens": 400,
                },
            )
        )
        policy.add_arm(
            Arm(
                name="B",
                params={
                    "k": k_b,
                    "context_budget_chars": ctx_b,
                    "mmr_lambda": mmr_b,
                    "temperature": temp_b,
                    "max_tokens": 400,
                },
            )
        )
        if add_arm_c:
            policy.add_arm(
                Arm(
                    name="C",
                    params={
                        "k": k_c,
                        "context_budget_chars": ctx_c,
                        "mmr_lambda": mmr_c,
                        "temperature": temp_c,
                        "max_tokens": 400,
                    },
                )
            )

        records: list[dict[str, Any]] = []
        for t in range(int(iters)):
            arm = policy.choose()
            q = random.choice(queries)

            arm_params = dict(arm.params)
            t0 = time.time()
            res = orch.run(q, arm_name=arm.name, arm_params=arm_params)
            elapsed_ms = (time.time() - t0) * 1000.0

            s = res.final_scores
            # grab tokens / latency from last critique
            last = res.timeline[-1] if res.timeline else {}
            critic_tokens = int(last.get("tokens") or 0)

            reward = compute_reward(s["overall"], critic_tokens, elapsed_ms)
            policy.update(arm.name, reward)

            records.append(
                {
                    "t": t,
                    "query": q,
                    "arm": arm.name,
                    "overall": s["overall"],
                    "faithfulness": s["faithfulness"],
                    "coverage": s["coverage"],
                    "clarity": s["clarity"],
                    "tokens": critic_tokens,
                    "latency_ms": elapsed_ms,
                    "reward": reward,
                }
            )

        df = pd.DataFrame.from_records(records)
        os.makedirs("data", exist_ok=True)
        out_csv = "data/rl_runs.csv"
        df.to_csv(out_csv, index=False)

        st.success(f"Completed {len(df)} iterations. Saved runs to {out_csv}")

        # ---- KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean Overall", f"{df['overall'].mean():.3f}")
        c2.metric("Mean Reward", f"{df['reward'].mean():.3f}")
        c3.metric("Mean Tokens", f"{df['tokens'].mean():.0f}")
        c4.metric("Mean Latency (ms)", f"{df['latency_ms'].mean():.0f}")

        st.subheader("Reward vs iteration")
        st.line_chart(df.set_index("t")[["reward"]])

        st.subheader("Arm selection share")
        freq = df["arm"].value_counts(normalize=True)
        st.bar_chart(freq)

        st.subheader("Quality vs tokens")
        st.scatter_chart(df[["tokens", "overall"]])

        st.subheader("Per-arm summary")
        by_arm = (
            df.groupby("arm")
            .agg(
                mean_overall=("overall", "mean"),
                mean_reward=("reward", "mean"),
                mean_tokens=("tokens", "mean"),
                mean_latency=("latency_ms", "mean"),
            )
            .reset_index()
        )
        st.dataframe(by_arm)

with tabs[3]:
    st.header("Evaluation")

    # Section 1: Bandit Policy Learning
    st.subheader("1. Bandit Policy Learning (RL Algorithm)")
    st.caption("Shows how the RL algorithm learns which configuration (arm) achieves best quality-efficiency tradeoff")
    rl_learning_dashboard(log_path="data/rl_bandit_rewards.csv")

    st.markdown("---")

    # Section 2: Offline Benchmark Results
    st.subheader("2. Offline Benchmark Results")
    st.caption("Comprehensive evaluation across 5 configurations with statistical analysis")

    # Offline benchmark PNGs
    qvt = Path("data/reports/quality_vs_tokens.png")
    cdf = Path("data/reports/latency_cdf.png")

    if qvt.exists():
        st.markdown("**Quality vs Tokens Tradeoff**")
        st.image(str(qvt))
    else:
        st.info("Run `python examples/evaluation/offline_eval.py` to generate quality_vs_tokens.png.")

    if cdf.exists():
        st.markdown("**Latency CDF**")
        st.image(str(cdf))

    st.markdown("---")

    # Section 3: Detailed RL Execution Logs
    st.subheader("3. RL Execution Details (Per-Query Logs)")
    st.caption("Detailed per-query results from RL experiments via API")
    rl_eval_section(csv_path="data/rl_runs_api.csv")


with tabs[4]:
    history_tab()
