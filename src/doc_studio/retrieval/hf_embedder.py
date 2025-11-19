from __future__ import annotations
import numpy as np, torch
from typing import List, Optional
from sentence_transformers import SentenceTransformer

def _dev():
    # pick what device torch can use quick
    if torch.cuda.is_available(): return "cuda"
    if getattr(torch.backends,"mps",None) and torch.backends.mps.is_available(): return "mps"
    return "cpu"

class HFEmbedder:
    """small helper wraps sentence transformers for fast embeddings"""
    def __init__(self, model_name:str="BAAI/bge-small-en-v1.5", device:Optional[str]=None, bs:int=64, norm:bool=True):
        self.dev = device or _dev()  # choose gpu/mps/cpu
        self.bs = bs; self.norm = norm
        self.m = SentenceTransformer(model_name, device=self.dev)

    def __call__(self, texts:List[str])->np.ndarray:
        # encode list of strings into float32 embeddings
        if not texts:
            # handle empty input safe
            d = getattr(self.m,"get_sentence_embedding_dimension",lambda:384)()
            return np.zeros((0,d),dtype=np.float32)
        x = self.m.encode(
            texts,
            batch_size=self.bs,
            convert_to_numpy=True,
            normalize_embeddings=self.norm,  # l2 norm if true
            show_progress_bar=False,
        )
        return x.astype(np.float32,copy=False)
