from agent_orch.retrieval import Document, TfidfRetriever

def main():
    retriever = TfidfRetriever()
    retriever.add([
        Document("d1", "LoRA and QLoRA are lightweight fine-tuning methods for LLMs."),
        Document("d2", "Reinforcement learning from human feedback (RLHF) aligns models."),
        Document("d3", "FlashAttention accelerates attention by IO-aware tiling on GPUs."),
        Document("d4", "Maximal Marginal Relevance (MMR) balances relevance and diversity in RAG."),
        Document("d5", "DPO is a preference optimization approach without explicit reward models."),
    ])
    print("Indexed:", retriever.size())

    q = "How does RAG reduce hallucinations using MMR?"
    hits = retriever.search(q, k=3)
    for i, h in enumerate(hits, 1):
        print(f"{i}. {h.doc.doc_id}  score={h.score:.3f}  text={h.doc.text}")

if __name__ == "__main__":
    main()
