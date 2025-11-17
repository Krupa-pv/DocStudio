import os
from doc_studio.core.config import Settings
from doc_studio.core.logging import setup_logging
from doc_studio.core.llm import LLMClient
from doc_studio.retrieval import Document, TfidfRetriever
from doc_studio.agents import SummarizerAgent
from doc_studio.retrieval.hf_embedder import HFEmbedder
from doc_studio.retrieval.faiss_retriever import FaissRetriever

def main():
    #uncomment below for no open ai api call for local tetsing:
    # os.environ["DRY_RUN"] = "1"

    cfg = Settings.load()
    setup_logging(cfg.log_level)

    """
    retriever = TfidfRetriever(ngram_range=(1, 2))
    retriever.add([
        Document("d1", "LoRA and QLoRA are lightweight fine-tuning methods for LLMs."),
        Document("d2", "Reinforcement learning from human feedback (RLHF) aligns models with human preferences."),
        Document("d3", "FlashAttention accelerates Transformer attention via IO-aware tiling on GPUs."),
        Document("d4", "Maximal Marginal Relevance (MMR) balances relevance and diversity for retrieval in RAG systems."),
        Document("d5", "DPO (Direct Preference Optimization) optimizes policies from preferences without an explicit reward model."),
    ])
    """

    emb = HFEmbedder("BAAI/bge-small-en-v1.5")
    retriever = FaissRetriever(emb)
    retriever.add([
        Document("d1", "LoRA and QLoRA are lightweight fine-tuning methods for LLMs."),
        Document("d2", "Reinforcement learning from human feedback (RLHF) aligns models with human preferences."),
        Document("d3", "FlashAttention accelerates Transformer attention via IO-aware tiling on GPUs."),
        Document("d4", "Maximal Marginal Relevance (MMR) balances relevance and diversity for retrieval in RAG systems."),
        Document("d5", "DPO (Direct Preference Optimization) optimizes policies from preferences without an explicit reward model."),
    ])
    client = LLMClient(cfg)
    agent = SummarizerAgent(retriever, client, k=4, context_budget_chars=1200,
                            use_mmr=True,
                            mmr_lambda=0.6,
                            embedding_fn=emb)

    query = "How does MMR help reduce hallucinations in RAG pipelines?"
    res = agent.summarize(query, temperature=0.0, max_tokens=250)

    print("-- Used docs ---")
    for doc_id, score in res.used_docs:
        print(f"{doc_id}  score={score:.3f}")

    print("\n--- Summary --")
    print(res.summary)

    print("\n-- LLM usage --")
    print(dict(
        prompt=res.llm.prompt_tokens,
        completion=res.llm.completion_tokens,
        total=res.llm.total_tokens,
        latency_ms=round(res.llm.latency_ms, 2),
    ))

if __name__ == "__main__":
    main()

