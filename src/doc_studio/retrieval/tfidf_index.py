from __future__ import annotations

from typing import Iterable, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .interfaces import Document, ScoredDocument, Retriever

"""
vectorizatoin happens by finding all unique words across all docs and assigning a position 
for each in unique word vector
--> 
represent each document as a vector with 1s and 0s in the position where that unique word 
is present or not respectively
--> 
TF-IDF weighting: 
term frequency (TF): how common word is
inverse document frequency (IDF): how unique word is across docs
"""

class TfidfRetriever(Retriever):
    """
    in-memory TF-IDF retriever
    
    """

    def __init__(self, ngram_range: tuple[int, int] = (1, 2), max_features: int | None = 5000):
        self._docs: list[Document] = []
        #vectorizer that converts text into TF-IDF vectors 
        
        self._vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            stop_words="english", #remove useless words 
        )
        self._matrix = None  

    def add(self, docs: Iterable[Document]) -> None:
        new_docs = list(docs)
        if not new_docs:
            return
        self._docs.extend(new_docs)

        
        corpus = [d.text for d in self._docs]
        mat = self._vectorizer.fit_transform(corpus)  #build the sparse matrix
        self._matrix = normalize(mat, norm="l2", copy=False)  #take l2 norm 

    def search(self, query: str, k: int = 5) -> List[ScoredDocument]:
        if not self._docs or self._matrix is None:
            return []

        q_vec = self._vectorizer.transform([query]) #turn query into TF-IDF vector 
        q_vec = normalize(q_vec, norm="l2", copy=False)

        # Cosine similarity = dot product (since rows are L2-normalized)
        sims = (self._matrix @ q_vec.T).toarray().ravel() 


        top_idx = np.argsort(-sims)[:k] #sort similarity scores and pick top k
        return [
            ScoredDocument(doc=self._docs[i], score=float(sims[i]))
            for i in top_idx
        ]

    def size(self) -> int:
        return len(self._docs)
