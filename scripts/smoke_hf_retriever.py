from agent_orch.retrieval import Document
from agent_orch.retrieval.hf_embeddings_retriever import HFEmbeddingsRetriever
from agent_orch.retrieval.hf_embedder import HFEmbedder

def main():
    r = HFEmbeddingsRetriever(HFEmbedder("sentence-transformers/all-MiniLM-L6-v2"))
    r.add([
        Document("d1", "RAG reduces hallucinations by grounding generation in retrieved evidence."),
        Document("d2", "DPO optimizes a policy directly from preferences without a reward model."),
        Document("d3", "MMR balances relevance and diversity to reduce redundancy in retrieval."),
    ])
    print("Indexed:", r.size())
    for h in r.search("How does MMR help retrieval?", k=2):
        print(h.doc.doc_id, f"{h.score:.3f}")

if __name__ == "__main__":
    main()
