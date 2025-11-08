from __future__ import annotations
import numpy as np
from typing import List, Callable
from .interfaces import ScoredDocument

def _l2n(X:np.ndarray)->np.ndarray:
    # quick normalize each row
    n=np.linalg.norm(X,axis=1,keepdims=True)+1e-12
    return X/n

class MMR:
    # reranks docs by relevance-diversity balance
    def __init__(self, embedding_fn:Callable[[list[str]],np.ndarray], lam:float=0.6):
        self.f=embedding_fn; self.lam=lam

    def rerank(self, query:str, cands:List[ScoredDocument], k:int)->List[ScoredDocument]:
        if not cands: return []
        txt=[c.doc.text for c in cands]
        Q=_l2n(self.f([query])); D=_l2n(self.f(txt))
        rel=(D@Q.T).ravel()  # cosine sim
        sel=[]; rem=list(range(len(cands)))
        while rem and len(sel)<k:
            if not sel:
                j=int(np.argmax(rel[rem])); sel.append(rem.pop(j)); continue
            S=D[sel]; sims=D[rem]@S.T; mx=sims.max(axis=1)
            score=self.lam*rel[rem]-(1-self.lam)*mx
            j=int(np.argmax(score)); sel.append(rem.pop(j))
        out=[ScoredDocument(cands[i].doc,float(rel[i])) for i in sel]
        return out
