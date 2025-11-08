from agent_orch.retrieval.hf_embedder import HFEmbedder
from agent_orch.retrieval.faiss_retriever import FaissRetriever

def main():
    emb=HFEmbedder("BAAI/bge-small-en-v1.5")
    r=FaissRetriever(emb)
    r.load("data/index.faiss", "data/index.meta.jsonl")
    print("loaded docs:", r.size())
    hits=r.search("quick sanity check on retrieval diversity", k=3)
    print([h.doc.doc_id for h in hits])

if __name__=="__main__":
    main()
