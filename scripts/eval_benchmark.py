from __future__ import annotations
import os, json, time, math, csv
from dataclasses import dataclass, asdict
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt

# ---- Project imports (adjust if your paths differ)
from doc_studio.core.config import Settings
from doc_studio.core.logging import setup_logging
from doc_studio.core.llm import LLMClient
from doc_studio.orchestrator.orchestrate import Orchestrator, OrchestratorConfig
from doc_studio.retrieval import Document, TfidfRetriever
from doc_studio.retrieval.hf_embedder import HFEmbedder
from doc_studio.retrieval.hf_embeddings_retriever import HFEmbeddingsRetriever

# ---------- Config ----------
REPORT_DIR = "data/reports"
EVAL_FILE = "data/eval.jsonl"          # [{"query": "..."}] (answer optional)
CORPUS_FILE = "data/corpus.jsonl"      # [{"id": "d1", "text": "..."}]
RESULTS_CSV = "data/eval_results.csv"
SEED = 7

# α/β used to compute reward-like score for reporting (your orchestrator already has its own)
ALPHA_PER_KTOK = 0.02       # penalty per 1k tokens
BETA_PER_SEC = 0.02         # latency penalty per second

np.random.seed(SEED)

@dataclass
class Row:
    config: str
    idx: int
    query: str
    overall: float
    faithfulness: float
    coverage: float
    clarity: float
    tokens: int
    latency_ms: float
    reward_like: float

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def ensure_eval_and_corpus():
    os.makedirs("data", exist_ok=True)
    # Minimal eval set if not present
    if not os.path.exists(EVAL_FILE):
        eval_items = [
            {"query": "Explain how MMR reduces hallucinations in RAG with citations."},
            {"query": "Contrast RLHF and DPO for alignment with citations."},
            {"query": "How does FlashAttention speed up attention on GPUs?"},
            {"query": "What are benefits of embeddings over TF-IDF for semantic search?"},
            {"query": "What is retrieval diversification and why does it matter?"},
        ]
        with open(EVAL_FILE, "w") as f:
            for it in eval_items:
                f.write(json.dumps(it) + "\n")

    # Minimal corpus if not present
    if not os.path.exists(CORPUS_FILE):
        corpus = [
            {"id": "d1", "text": "LoRA and QLoRA are lightweight fine-tuning methods for LLMs."},
            {"id": "d2", "text": "Reinforcement learning from human feedback (RLHF) aligns models with human preferences using a reward model."},
            {"id": "d3", "text": "FlashAttention accelerates Transformer attention via IO-aware tiling on GPUs."},
            {"id": "d4", "text": "Maximal Marginal Relevance (MMR) balances relevance and diversity for retrieval in RAG systems, reducing redundancy and hallucinations."},
            {"id": "d5", "text": "DPO optimizes policies directly from preferences without an explicit reward model."},
            {"id": "d6", "text": "Embeddings capture semantic similarity; TF-IDF uses lexical overlap. Embeddings usually improve recall for paraphrased queries."},
            {"id": "d7", "text": "Diversification in retrieval ensures broader coverage of aspects, improving grounding and reducing overfitting to one source."},
        ]
        with open(CORPUS_FILE, "w") as f:
            for it in corpus:
                f.write(json.dumps(it) + "\n")

def build_retriever_tf_idf(docs: List[Document]):
    r = TfidfRetriever()
    r.add(docs)
    return r

def build_retriever_hf(docs: List[Document], model_name: str = "BAAI/bge-small-en-v1.5"):
    emb = HFEmbedder(model_name)
    r = HFEmbeddingsRetriever(emb)
    r.add(docs)
    return r, emb

def run_config(
    name: str,
    retriever,
    llm_client: LLMClient,
    queries: List[str],
    summarizer_knobs: Dict[str, Any],
) -> List[Row]:
    """
    summarizer_knobs may include:
      k, temperature, max_tokens, context_budget_chars, use_mmr, mmr_lambda, embedder (callable)
    """
    cfg = OrchestratorConfig(quality_threshold=0.80, max_attempts=2, use_rewrite_on_retry=True)
    orch = Orchestrator(retriever=retriever, llm=llm_client, cfg=cfg, reward_log_path="data/rewards.csv")

    rows: List[Row] = []
    for i, q in enumerate(queries):
        t0 = time.time()
        res = orch.run(query=q, arm_name=name, arm_params=summarizer_knobs)
        latency_ms = (time.time() - t0) * 1000.0

        # Pull tokens from the critic step if present; otherwise 0
        # (You can extend this to include summarizer tokens if your result exposes them.)
        critic_tokens = 0
        if res.timeline:
            last = res.timeline[-1]
            critic_tokens = int(last.get("critic_tokens") or 0)

        s = res.final_scores
        # Compute a display reward-like metric as you described
        reward_like = s["overall"] - ALPHA_PER_KTOK * (critic_tokens / 1000.0) - BETA_PER_SEC * (latency_ms / 1000.0)

        rows.append(Row(
            config=name, idx=i, query=q,
            overall=float(s["overall"]),
            faithfulness=float(s["faithfulness"]),
            coverage=float(s["coverage"]),
            clarity=float(s["clarity"]),
            tokens=critic_tokens,
            latency_ms=latency_ms,
            reward_like=float(reward_like),
        ))
    return rows

def summarize(rows: List[Row]) -> Dict[str, float]:
    def mean(xs): return float(np.mean(xs)) if xs else 0.0
    return {
        "overall": mean([r.overall for r in rows]),
        "faithfulness": mean([r.faithfulness for r in rows]),
        "coverage": mean([r.coverage for r in rows]),
        "clarity": mean([r.clarity for r in rows]),
        "tokens": mean([r.tokens for r in rows]),
        "latency_ms": mean([r.latency_ms for r in rows]),
        "reward_like": mean([r.reward_like for r in rows]),
        "n": len(rows),
    }

def pct_delta(new: float, base: float) -> float:
    return 0.0 if base == 0 else 100.0 * (new - base) / base

def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    ensure_eval_and_corpus()

    # Load eval set & corpus
    eval_items = load_jsonl(EVAL_FILE)
    queries = [it["query"] for it in eval_items]
    corpus = [Document(it["id"], it["text"]) for it in load_jsonl(CORPUS_FILE)]

    # Setup system
    settings = Settings.load()
    setup_logging("INFO")
    llm = LLMClient(settings)

    # ---- Build retrievers
    tfidf_ret = build_retriever_tf_idf(corpus)
    hf_ret, hf_embedder = build_retriever_hf(corpus, model_name="BAAI/bge-small-en-v1.5")

    # ---- Define 3 configs
    configs = [
        ("baseline_tfidf", tfidf_ret, {"k": 5, "temperature": 0.0, "max_tokens": 300, "context_budget_chars": 3000, "use_mmr": False}),
        ("embeddings_only", hf_ret, {"k": 5, "temperature": 0.0, "max_tokens": 300, "context_budget_chars": 3000, "use_mmr": False}),
        ("embeddings_mmr",  hf_ret, {"k": 5, "temperature": 0.0, "max_tokens": 300, "context_budget_chars": 3000, "use_mmr": True, "mmr_lambda": 0.6, "embedder": hf_embedder}),
    ]

    # ---- Run
    all_rows: List[Row] = []
    for name, retr, knobs in configs:
        print(f"\n>> Running config: {name}")
        rows = run_config(name, retr, llm, queries, knobs)
        all_rows.extend(rows)
        s = summarize(rows)
        print(f"   mean overall={s['overall']:.3f}  faith={s['faithfulness']:.3f}  cov={s['coverage']:.3f}  "
              f"clar={s['clarity']:.3f}  tok={s['tokens']:.1f}  lat={s['latency_ms']:.0f}ms  reward≈{s['reward_like']:.3f}")

    # ---- Save results CSV
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(all_rows[0]).keys()))
        w.writeheader()
        for r in all_rows:
            w.writerow(asdict(r))
    print(f"\nSaved per-run results to {RESULTS_CSV}")

    # ---- Aggregate by config and print table with deltas vs baseline
    by_cfg: Dict[str, List[Row]] = {}
    for r in all_rows:
        by_cfg.setdefault(r.config, []).append(r)
    agg = {cfg: summarize(rows) for cfg, rows in by_cfg.items()}

    base = agg["baseline_tfidf"]
    def fmt_row(name, a):
        return [
            name,
            f"{a['overall']:.3f} ({pct_delta(a['overall'], base['overall']):+5.1f}%)",
            f"{a['faithfulness']:.3f}",
            f"{a['coverage']:.3f}",
            f"{a['tokens']:.0f} ({pct_delta(a['tokens'], base['tokens']):+5.1f}%)",
            f"{a['latency_ms']:.0f}ms ({pct_delta(a['latency_ms'], base['latency_ms']):+5.1f}%)",
            f"{a['reward_like']:.3f} ({pct_delta(a['reward_like'], base['reward_like']):+5.1f}%)",
            int(a["n"]),
        ]

    headers = ["Config", "Overall (Δ%)", "Faith", "Cover", "Tokens (Δ%)", "Latency (Δ%)", "Reward≈ (Δ%)", "n"]
    table = [fmt_row(name, agg[name]) for name in ["baseline_tfidf","embeddings_only","embeddings_mmr"]]
    colw = [max(len(str(x)) for x in col) for col in zip(headers, *table)]

    print("\n=== Aggregate Results ===")
    print(" | ".join(h.ljust(colw[i]) for i, h in enumerate(headers)))
    print("-+-".join("-"*w for w in colw))
    for row in table:
        print(" | ".join(str(row[i]).ljust(colw[i]) for i in range(len(headers))))

    # ---- Plots
    # Quality vs Tokens (scatter, colored by config)
    colors = {"baseline_tfidf":"tab:gray","embeddings_only":"tab:blue","embeddings_mmr":"tab:green"}
    plt.figure()
    for cfg, rows in by_cfg.items():
        x = [r.tokens for r in rows]
        y = [r.overall for r in rows]
        plt.scatter(x, y, label=cfg, alpha=0.75)
    plt.xlabel("Tokens (critic; approx)")
    plt.ylabel("Overall quality")
    plt.title("Quality vs Tokens")
    plt.legend()
    out1 = os.path.join(REPORT_DIR, "quality_vs_tokens.png")
    plt.savefig(out1, bbox_inches="tight", dpi=150)
    plt.close()

    # Latency CDF
    plt.figure()
    for cfg, rows in by_cfg.items():
        xs = np.sort([r.latency_ms for r in rows])
        ys = np.linspace(0, 1, len(xs), endpoint=True)
        plt.plot(xs, ys, label=cfg)
    plt.xlabel("Latency (ms)")
    plt.ylabel("CDF")
    plt.title("Latency CDF")
    plt.legend()
    out2 = os.path.join(REPORT_DIR, "latency_cdf.png")
    plt.savefig(out2, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"\nSaved plots:\n  {out1}\n  {out2}")
    print("\nDone.")

if __name__ == "__main__":
    main()
