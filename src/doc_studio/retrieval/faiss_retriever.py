from __future__ import annotations
import json
from pathlib import Path
import faiss, numpy as np
from typing import Iterable, List, Optional
from .hf_embedder import HFEmbedder
from .interfaces import Document, ScoredDocument, Retriever

class FaissRetriever(Retriever):
    """ uses faiss for similarity search, inner product == cosine if normed"""
    def __init__(self, emb:Optional[HFEmbedder]=None, dim:int|None=None):
        self.emb = emb or HFEmbedder()
        if dim is None: dim = self.emb(["_tmp_"]).shape[1]
        self.dim = dim
        self.idx = faiss.IndexFlatIP(self.dim)  # flat --> exact search so no approximation 
                                                # IP = inner product (cos similarty)
        self.docs:list[Document] = []
        self._E:np.ndarray|None=None

    def add(self, docs:Iterable[Document])->None:
        # add list of docs and rebuild index (simple readd)
        ds=list(docs)
        if not ds: return
        self.docs.extend(ds)
        corpus=[d.text for d in self.docs]
        E=self.emb(corpus).astype(np.float32)
        self._E=E
        self.idx.reset(); self.idx.add(E)

    def search(self,q:str,k:int=5)->List[ScoredDocument]:
        # embed query then get top k similar docs
        if not self.docs or self._E is None or self.idx.ntotal==0: return []
        Q=self.emb([q]).astype(np.float32)
        sims,ids=self.idx.search(Q,k)
        out=[]
        for i,s in zip(ids[0],sims[0]):
            if i==-1: continue
            out.append(
                ScoredDocument(
                    doc=self.docs[i],
                    score=float(s),
                    meta={
                        "rank": len(out),
                        "raw_score": float(s),
                        "retriever": "faiss",
                    }
                )
            )

        return out

    def size(self)->int: return len(self.docs)

    def save(self, p_idx: str | Path, p_meta: str | Path) -> None:
        """Save index file + meta lines."""
        p_idx = str(p_idx)
        p_meta = str(p_meta)

        faiss.write_index(self.idx, p_idx)

        with open(p_meta, "w", encoding="utf-8") as f:
            for d in self.docs:
                f.write(
                    json.dumps(
                        {
                            "doc_id": d.doc_id,
                            "text": d.text,
                            "meta": d.meta,
                        }
                    )
                    + "\n"
                )

    def load(self, p_idx: str | Path, p_meta: str | Path) -> None:
        """Load index and docs from disk."""
        p_idx = str(p_idx)
        p_meta = str(p_meta)

        self.idx = faiss.read_index(p_idx)
        self.docs.clear()

        with open(p_meta, "r", encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                self.docs.append(Document(o["doc_id"], o["text"], o.get("meta")))

        corpus = [d.text for d in self.docs]
        if corpus:
            self._E = self.emb(corpus).astype(np.float32)
        else:
            self._E = None
