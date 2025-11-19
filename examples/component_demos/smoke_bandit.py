import random
from doc_studio.core.config import Settings
from doc_studio.core.logging import setup_logging
from doc_studio.core.llm import LLMClient
from doc_studio.retrieval import Document, TfidfRetriever
from doc_studio.orchestrator.simple import Orchestrator, OrchestratorConfig
from doc_studio.orchestrator.policy import EpsilonGreedyPolicy, Arm

def main():
    cfg = Settings.load()
    setup_logging(cfg.log_level)

    # Tiny corpus as before
    retriever = TfidfRetriever()
    retriever.add([
        Document("d1", "LoRA and QLoRA are lightweight fine-tuning methods for LLMs."),
        Document("d2", "Reinforcement learning from human feedback (RLHF) aligns models with human preferences."),
        Document("d3", "FlashAttention accelerates Transformer attention via IO-aware tiling on GPUs."),
        Document("d4", "Maximal Marginal Relevance (MMR) balances relevance and diversity for retrieval in RAG systems, which can reduce hallucinations."),
        Document("d5", "DPO optimizes policies from preferences without an explicit reward model."),
    ])

    client = LLMClient(cfg)

    # Define a few arms (summarizer/critic knobs)
    arms = [
        Arm("cheap_default", {"k": 3, "temperature": 0.0, "context_budget_chars": 1800,
                            "use_mmr": True, "mmr_lambda": 0.6, "embedder": emb, "max_tokens": 250}),
        Arm("balanced_quality", {"k": 5, "temperature": 0.0, "context_budget_chars": 2800,
                                "use_mmr": True, "mmr_lambda": 0.6, "embedder": emb, "max_tokens": 300}),
        Arm("high_coverage", {"k": 7, "temperature": 0.2, "context_budget_chars": 3600,
                            "use_mmr": True, "mmr_lambda": 0.7, "embedder": emb, "max_tokens": 350}),
    ]


    policy = EpsilonGreedyPolicy(arms=arms, epsilon=0.25, reward_log_path="data/rewards.csv")
    orch = Orchestrator(retriever=retriever, llm=client, cfg=OrchestratorConfig(quality_threshold=0.80, max_attempts=2), reward_log_path="data/rewards.csv")

    queries = [
        "Explain how MMR reduces hallucinations in RAG with citations.",
        "What is the role of RLHF vs DPO in post-training alignment?",
        "How does FlashAttention accelerate transformers on GPUs?",
    ]

    # Run a few iterations (simulating usage)
    for i in range(5):
        arm = policy.choose()
        q = random.choice(queries)
        res = orch.run(query=q, arm_name=arm.name, arm_params=arm.params)
        print(f"[iter {i+1}] arm={arm.name} overall={res.final_scores['overall']:.3f} reward={res.final_scores['reward']:.3f}")

    print("Bandit runs complete. Check data/rewards.csv for per-arm rewards.")

if __name__ == "__main__":
    main()
