from doc_studio.core.config import Settings
from doc_studio.core.logging import setup_logging
from doc_studio.core.llm import LLMClient
from doc_studio.retrieval import Document, TfidfRetriever
from doc_studio.agents import SummarizerAgent
from doc_studio.agents.analyst import AnalystAgent

def main():
    cfg = Settings.load()
    setup_logging(cfg.log_level)

    # Tiny corpus
    retriever = TfidfRetriever()
    retriever.add([
        Document("d1", "LoRA and QLoRA are lightweight fine-tuning methods for LLMs."),
        Document("d2", "Reinforcement learning from human feedback (RLHF) aligns models with human preferences."),
        Document("d3", "FlashAttention accelerates Transformer attention via IO-aware tiling on GPUs."),
        Document("d4", "Maximal Marginal Relevance (MMR) balances relevance and diversity for retrieval in RAG systems, which can reduce hallucinations by providing diverse, relevant evidence."),
        Document("d5", "DPO optimizes policies from preferences without an explicit reward model."),
    ])

    client = LLMClient(cfg)
    summarizer = SummarizerAgent(retriever, client, k=4, context_budget_chars=1200)
    analyst = AnalystAgent(client, do_rewrite=True)

    query = "How does MMR help reduce hallucinations in RAG pipelines?"
    sum_res = summarizer.summarize(query, temperature=0.0, max_tokens=220)

    print("\n--- Summary--")
    print(sum_res.summary)

    crit = analyst.critique(query=query, hits=retriever.search(query, k=4), summary=sum_res.summary)

    print("\n---- Scores --")
    print(vars(crit.scores))

    print("\n-- Justification --")
    print(crit.justification)

    if crit.suggested_rewrite:
        print("\n---- Suggested rewrite --")
        print(crit.suggested_rewrite)

    print("\n-- LLM usage (critic) ---")
    print(dict(
        prompt=crit.llm.prompt_tokens,
        completion=crit.llm.completion_tokens,
        total=crit.llm.total_tokens,
        latency_ms=round(crit.llm.latency_ms, 2),
    ))

if __name__ == "__main__":
    main()
