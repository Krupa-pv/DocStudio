from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from doc_studio.core.config import Settings
from doc_studio.core.logging import setup_logging
from doc_studio.core.llm import LLMClient
from doc_studio.retrieval.faiss_retriever import FaissRetriever
from doc_studio.retrieval.hf_embedder import HFEmbedder
from doc_studio.orchestrator.orchestrate import Orchestrator, OrchestratorConfig


@dataclass
class EvalRow:
    """Single evaluation run result with all tracked metrics"""
    config: str
    query: str
    overall: float
    faithfulness: float
    coverage: float
    clarity: float
    reward: float
    tokens_prompt: int
    tokens_completion: int
    tokens_total: int
    latency_ms: float
    attempts: int


@dataclass
class ConfigStats:
    """Aggregated statistics for a configuration across all queries"""
    config: str
    n_queries: int

    # Quality metrics (higher is better)
    overall_mean: float
    overall_std: float
    overall_median: float
    overall_p95: float

    faithfulness_mean: float
    coverage_mean: float
    clarity_mean: float

    # Efficiency metrics (lower is better for tokens/latency)
    tokens_mean: float
    tokens_std: float
    tokens_median: float
    tokens_p95: float

    latency_mean: float
    latency_std: float
    latency_median: float
    latency_p95: float

    # Combined metric
    reward_mean: float
    reward_std: float

    # Process metrics
    attempts_mean: float
    retry_rate: float


# Configuration definitions for ablation study
# Each config tests a specific hypothesis about RAG performance
ARM_CONFIGS: Dict[str, Dict[str, Any]] = {
    "baseline_tfidf": {
        "k": 6,
        "context_budget_chars": 4000,
        "mmr_lambda": 0.0,  # No diversity reranking
        "temperature": 0.2,
        "max_tokens": 600,
        "use_mmr": False,
        "retriever_type": "tfidf",  # For comparison if implemented
    },
    "embeddings_only": {
        "k": 5,
        "context_budget_chars": 3000,
        "mmr_lambda": 0.0,  # Pure relevance, no diversity
        "temperature": 0.0,
        "max_tokens": 500,
        "use_mmr": False,
    },
    "embeddings_mmr_balanced": {
        "k": 5,
        "context_budget_chars": 3000,
        "mmr_lambda": 0.6,  # Balance relevance and diversity
        "temperature": 0.0,
        "max_tokens": 500,
        "use_mmr": True,
    },
    "efficient_mmr_high": {
        "k": 4,
        "context_budget_chars": 2200,
        "mmr_lambda": 0.7,  # Higher diversity
        "temperature": 0.0,
        "max_tokens": 450,
        "use_mmr": True,
    },
    "aggressive_efficiency": {
        "k": 3,
        "context_budget_chars": 1400,
        "mmr_lambda": 0.6,
        "temperature": 0.0,
        "max_tokens": 350,
        "use_mmr": True,
    },
}


def load_eval_queries(path: Path) -> List[str]:
    """Load evaluation queries from JSONL file"""
    if not path.exists():
        raise FileNotFoundError(f"Eval file not found: {path}")
    queries: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = obj.get("query") or obj.get("question")
            if q:
                queries.append(q)
    if not queries:
        raise RuntimeError(f"No queries found in {path}")
    return queries


def build_orchestrator() -> Orchestrator:
    """Initialize orchestrator with FAISS retriever and configured LLM"""
    cfg = Settings.load()
    setup_logging(getattr(cfg, "log_level", "INFO"))

    llm = LLMClient(cfg)
    emb = HFEmbedder("BAAI/bge-small-en-v1.5")
    ret = FaissRetriever(emb)

    # Load existing FAISS index or fail fast
    idx_path = Path("data/index.faiss")
    meta_path = Path("data/index.meta.jsonl")
    if idx_path.exists() and meta_path.exists():
        ret.load(str(idx_path), str(meta_path))
    else:
        raise RuntimeError(
            "FAISS index not found at data/index.faiss. "
            "Run data preparation scripts before evaluation"
        )

    orch = Orchestrator(
        retriever=ret,
        llm=llm,
        emb_fn=emb,
        cfg=OrchestratorConfig(
            quality_threshold=0.75,
            max_attempts=2,
            use_rewrite_on_retry=True,
        ),
    )
    return orch


def run_offline_eval(
    eval_file: Path,
    out_csv: Path,
    seed: int = 7,
    configs: Dict[str, Dict[str, Any]] = ARM_CONFIGS,
) -> List[EvalRow]:
    """
    Run offline evaluation across all configs and queries
    Returns per-run results for detailed analysis
    """
    np.random.seed(seed)

    queries = load_eval_queries(eval_file)
    orch = build_orchestrator()

    rows: List[EvalRow] = []

    for cfg_name, arm_params in configs.items():
        print(f"\n=== Config: {cfg_name} ===")
        print(f"    Params: k={arm_params.get('k')}, "
              f"mmr_lambda={arm_params.get('mmr_lambda')}, "
              f"context={arm_params.get('context_budget_chars')}")

        for i, q in enumerate(queries, 1):
            print(f"  [{i}/{len(queries)}] {q[:60]}...", end="", flush=True)
            t0 = time.time()
            res = orch.run(query=q, arm_name=cfg_name, arm_params=arm_params)
            elapsed_ms = (time.time() - t0) * 1000.0

            scores = res.final_scores

            # Extract token counts from orchestrator result
            # This assumes OrchestratorResult has total_tokens attribute
            total_tokens = getattr(res, "total_tokens", 0)

            row = EvalRow(
                config=cfg_name,
                query=q,
                overall=float(scores.get("overall", 0.0)),
                faithfulness=float(scores.get("faithfulness", 0.0)),
                coverage=float(scores.get("coverage", 0.0)),
                clarity=float(scores.get("clarity", 0.0)),
                reward=float(scores.get("reward", 0.0)),
                tokens_prompt=0,  # Could extract from res if available
                tokens_completion=0,  # Could extract from res if available
                tokens_total=total_tokens,
                latency_ms=elapsed_ms,
                attempts=res.attempts,
            )
            rows.append(row)
            print(f" done (overall={row.overall:.3f}, tokens={row.tokens_total}, "
                  f"latency={row.latency_ms:.0f}ms)")

    # Write detailed CSV for further analysis
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

    return rows


def compute_statistics(rows: List[EvalRow]) -> Dict[str, ConfigStats]:
    """
    Compute comprehensive statistics per config
    Includes mean, std, median, p95 for all key metrics
    """
    by_cfg: Dict[str, List[EvalRow]] = {}
    for r in rows:
        by_cfg.setdefault(r.config, []).append(r)

    def safe_percentile(vals: List[float], p: int) -> float:
        """Compute percentile with fallback for small samples"""
        if not vals:
            return math.nan
        if len(vals) == 1:
            return vals[0]
        return float(np.percentile(vals, p))

    stats: Dict[str, ConfigStats] = {}

    for cfg_name, cfg_rows in by_cfg.items():
        overall_vals = [r.overall for r in cfg_rows]
        faith_vals = [r.faithfulness for r in cfg_rows]
        cover_vals = [r.coverage for r in cfg_rows]
        clarity_vals = [r.clarity for r in cfg_rows]
        reward_vals = [r.reward for r in cfg_rows]
        tokens_vals = [r.tokens_total for r in cfg_rows]
        latency_vals = [r.latency_ms for r in cfg_rows]
        attempts_vals = [r.attempts for r in cfg_rows]

        retry_count = sum(1 for a in attempts_vals if a > 1)
        retry_rate = retry_count / len(attempts_vals) if attempts_vals else 0.0

        stats[cfg_name] = ConfigStats(
            config=cfg_name,
            n_queries=len(cfg_rows),

            overall_mean=float(np.mean(overall_vals)),
            overall_std=float(np.std(overall_vals)),
            overall_median=float(np.median(overall_vals)),
            overall_p95=safe_percentile(overall_vals, 95),

            faithfulness_mean=float(np.mean(faith_vals)),
            coverage_mean=float(np.mean(cover_vals)),
            clarity_mean=float(np.mean(clarity_vals)),

            tokens_mean=float(np.mean(tokens_vals)),
            tokens_std=float(np.std(tokens_vals)),
            tokens_median=float(np.median(tokens_vals)),
            tokens_p95=safe_percentile(tokens_vals, 95),

            latency_mean=float(np.mean(latency_vals)),
            latency_std=float(np.std(latency_vals)),
            latency_median=float(np.median(latency_vals)),
            latency_p95=safe_percentile(latency_vals, 95),

            reward_mean=float(np.mean(reward_vals)),
            reward_std=float(np.std(reward_vals)),

            attempts_mean=float(np.mean(attempts_vals)),
            retry_rate=retry_rate,
        )

    return stats


def print_summary_table(stats: Dict[str, ConfigStats], baseline: str = "baseline_tfidf") -> None:
    """
    Print formatted summary table with delta percentages vs baseline
    This produces resume-ready metrics
    """
    print("\n" + "="*120)
    print("EVALUATION RESULTS - CONFIGURATION COMPARISON")
    print("="*120)

    print(f"\n{'Config':<25} | {'Overall':>8} | {'Reward':>8} | {'Tokens':>8} | "
          f"{'Latency(ms)':>12} | {'n':>3} | {'Retry%':>6}")
    print("-" * 120)

    baseline_stats = stats.get(baseline)

    for cfg_name, st in stats.items():
        if cfg_name == baseline or baseline_stats is None:
            # Print baseline without deltas
            print(f"{cfg_name:<25} | "
                  f"{st.overall_mean:>8.3f} | "
                  f"{st.reward_mean:>8.3f} | "
                  f"{st.tokens_mean:>8.0f} | "
                  f"{st.latency_mean:>12.1f} | "
                  f"{st.n_queries:>3} | "
                  f"{st.retry_rate*100:>5.1f}%")
        else:
            # Print with deltas vs baseline
            def pct_delta(val: float, base: float) -> str:
                if base == 0:
                    return "N/A"
                delta = 100.0 * (val - base) / base
                sign = "+" if delta >= 0 else ""
                return f"{sign}{delta:.1f}%"

            q_delta = pct_delta(st.overall_mean, baseline_stats.overall_mean)
            r_delta = pct_delta(st.reward_mean, baseline_stats.reward_mean)
            t_delta = pct_delta(st.tokens_mean, baseline_stats.tokens_mean)
            l_delta = pct_delta(st.latency_mean, baseline_stats.latency_mean)

            print(f"{cfg_name:<25} | "
                  f"{st.overall_mean:>8.3f} | "
                  f"{st.reward_mean:>8.3f} | "
                  f"{st.tokens_mean:>8.0f} | "
                  f"{st.latency_mean:>12.1f} | "
                  f"{st.n_queries:>3} | "
                  f"{st.retry_rate*100:>5.1f}%")
            print(f"{'':<25} | "
                  f"{q_delta:>8} | "
                  f"{r_delta:>8} | "
                  f"{t_delta:>8} | "
                  f"{l_delta:>12} |     |      ")

    print("="*120)


def print_detailed_stats(stats: Dict[str, ConfigStats]) -> None:
    """Print detailed statistics including std, median, p95 for deep analysis"""
    print("\n" + "="*120)
    print("DETAILED STATISTICS PER CONFIGURATION")
    print("="*120)

    for cfg_name, st in stats.items():
        print(f"\nConfig: {cfg_name}")
        print(f"  Queries: {st.n_queries}")
        print(f"  Overall Quality:")
        print(f"    Mean:   {st.overall_mean:.3f} (±{st.overall_std:.3f})")
        print(f"    Median: {st.overall_median:.3f}")
        print(f"    P95:    {st.overall_p95:.3f}")
        print(f"  Component Scores:")
        print(f"    Faithfulness: {st.faithfulness_mean:.3f}")
        print(f"    Coverage:     {st.coverage_mean:.3f}")
        print(f"    Clarity:      {st.clarity_mean:.3f}")
        print(f"  Tokens:")
        print(f"    Mean:   {st.tokens_mean:.0f} (±{st.tokens_std:.0f})")
        print(f"    Median: {st.tokens_median:.0f}")
        print(f"    P95:    {st.tokens_p95:.0f}")
        print(f"  Latency (ms):")
        print(f"    Mean:   {st.latency_mean:.1f} (±{st.latency_std:.1f})")
        print(f"    Median: {st.latency_median:.1f}")
        print(f"    P95:    {st.latency_p95:.1f}")
        print(f"  Reward:")
        print(f"    Mean:   {st.reward_mean:.3f} (±{st.reward_std:.3f})")
        print(f"  Process:")
        print(f"    Avg Attempts: {st.attempts_mean:.2f}")
        print(f"    Retry Rate:   {st.retry_rate*100:.1f}%")


def save_aggregate_stats(stats: Dict[str, ConfigStats], out_path: Path) -> None:
    """Save aggregated statistics to CSV for plotting and analysis"""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(asdict(next(iter(stats.values()))).keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for st in stats.values():
            writer.writerow(asdict(st))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline evaluation for DocStudio with comprehensive metrics"
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=Path("data/eval.jsonl"),
        help="Path to eval queries JSONL",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("data/eval_results.csv"),
        help="Output path for per-run results",
    )
    parser.add_argument(
        "--stats-csv",
        type=Path,
        default=Path("data/eval_stats.csv"),
        help="Output path for aggregated statistics",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="baseline_tfidf",
        help="Name of baseline config for delta calculations",
    )
    args = parser.parse_args()

    print("Starting offline evaluation...")
    print(f"Seed: {args.seed}")
    print(f"Configs: {len(ARM_CONFIGS)}")

    # Run evaluation
    rows = run_offline_eval(args.eval_file, args.out_csv, seed=args.seed)

    # Compute statistics
    stats = compute_statistics(rows)

    # Print results
    print_summary_table(stats, baseline=args.baseline)
    print_detailed_stats(stats)

    # Save outputs
    save_aggregate_stats(stats, args.stats_csv)

    print(f"\nPer-run results saved to: {args.out_csv}")
    print(f"Aggregate statistics saved to: {args.stats_csv}")
    print("\nUse these metrics for:")
    print("  - Resume bullets (% improvements over baseline)")
    print("  - Plotting quality vs efficiency tradeoffs")
    print("  - Demonstrating evaluation rigor")


if __name__ == "__main__":
    main()
