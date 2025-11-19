from __future__ import annotations
import json
from pathlib import Path

from doc_studio.retrieval.hf_embedder import HFEmbedder
from doc_studio.retrieval.faiss_retriever import FaissRetriever
from doc_studio.retrieval import Document


def load_corpus(path: Path):
    docs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            did = obj.get("id") or obj.get("doc_id")
            txt = obj.get("text") or obj.get("content")
            meta = obj.get("meta", {})
            if did and txt:
                docs.append(Document(did, txt, meta))
    return docs


def main():
    corpus_path = Path("data/corpus.jsonl")
    if not corpus_path.exists():
        raise FileNotFoundError("data/corpus.jsonl not found")

    docs = load_corpus(corpus_path)
    print(f"Loaded {len(docs)} docs")

    emb = HFEmbedder("BAAI/bge-small-en-v1.5")
    ret = FaissRetriever(emb)

    ret.add(docs)

    out_idx = Path("data/index.faiss")
    out_meta = Path("data/index.meta.jsonl")

    ret.save(out_idx, out_meta)
    print("Saved index to:", out_idx)
    print("Saved meta to:", out_meta)


if __name__ == "__main__":
    main()
