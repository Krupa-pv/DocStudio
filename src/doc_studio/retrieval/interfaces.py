from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Protocol, Iterable


@dataclass
class Document:
    doc_id: str
    text: str
    meta: dict | None = None


@dataclass
class ScoredDocument:
    doc: Document
    score: float
    meta:  dict[str, Any] | None = None


class Retriever(Protocol):
    def add(self, docs: Iterable[Document]) -> None: ...
    def search(self, query: str, k: int = 5) -> List[ScoredDocument]: ...
    def size(self) -> int: ...


