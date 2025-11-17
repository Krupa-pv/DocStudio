from doc_studio.retrieval import Document
from doc_studio.retrieval.hf_embedder import HFEmbedder
from doc_studio.retrieval.faiss_retriever import FaissRetriever
from doc_studio.retrieval.mmr import MMR

def main():
    emb=HFEmbedder("BAAI/bge-small-en-v1.5")
    r=FaissRetriever(emb)
    r.add([
        Document("d1","rag reduces hallucinations by grounding generation in retrieved evidence."),
        Document("d2","mmr balances relevance and diversity to reduce redundancy in retrieval for rag."),
        Document("d3","dpo optimizes policy directly from preferences no reward model."),
        Document("d4","flashattention accelerates transformer attention via io-aware tiling."),
        Document("d5","hybrid retrieval mixes bm25 with dense embeddings for recall."),
    ])
    print("indexed:",r.size())
    q="how does mmr help rag reduce hallucinations"
    base=r.search(q,k=5)
    print("base order:",[c.doc.doc_id for c in base])
    mmr=MMR(embedding_fn=emb,lam=0.6)
    rr=mmr.rerank(q,base,k=3)
    print("mmr order:",[c.doc.doc_id for c in rr])

if __name__=="__main__":
    main()
