from doc_studio.core.config import Settings
from doc_studio.core.logging import setup_logging
from doc_studio.core.llm import LLMClient
from doc_studio.retrieval import Document, TfidfRetriever
from doc_studio.retrieval.hf_embedder import HFEmbedder
from doc_studio.retrieval.faiss_retriever import FaissRetriever
from doc_studio.orchestrator.orchestrate import Orchestrator, OrchestratorConfig

def main():
    cfg = Settings.load()
    setup_logging(cfg.log_level)

    #retriever = TfidfRetriever()
    # embeddings + faiss (cosine via inner product on L2-normalized vectors)
    emb = HFEmbedder("BAAI/bge-small-en-v1.5")
    retriever = FaissRetriever(emb)

    retriever.add([
        Document("d1", "LoRA and QLoRA are lightweight fine-tuning methods for LLMs."),
        Document("d2", "Reinforcement learning from human feedback (RLHF) aligns models with human preferences."),
        Document("d3", "FlashAttention accelerates Transformer attention via IO-aware tiling on GPUs."),
        Document("d4", "Maximal Marginal Relevance (MMR) balances relevance and diversity for retrieval in RAG systems, which can reduce hallucinations by providing diverse, relevant evidence."),
        Document("d5", "DPO optimizes policies from preferences without an explicit reward model."),
    ])

    client = LLMClient(cfg)
    orch = Orchestrator(
        retriever=retriever,
        llm=None,
        emb_fn=emb,
        cfg=OrchestratorConfig(quality_threshold=0.80, max_attempts=2, use_rewrite_on_retry=True),
        reward_log_path="data/rewards.csv",
    )

    query = "Explain how MMR reduces hallucinations in RAG. Provide citations."
    res = orch.run(query)

    print("\n=== ORCHESTRATION RESULT ===")
    print("Attempts:", res.attempts)
    print("Final overall score:", round(res.final_scores["overall"], 3))
    print("Reward:", round(res.final_scores["reward"], 3))
    print("\nFinal summary:\n", res.final_summary)

    print("\nTimeline steps:")
    for step in res.timeline:
        print("-", step["step"], "overall=", round(step["scores"]["overall"], 3))

    print("\nReward log written to data/rewards.csv")

if __name__ == "__main__":
    main()
