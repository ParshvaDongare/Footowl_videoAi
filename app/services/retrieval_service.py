from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any

from app.rag.seeds import REMOTION_DOCS, STYLE_GUIDES
from app.utils import tokenize


def _vectorize(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokenize(text):
        counts[token] = counts.get(token, 0) + 1
    return counts


def _cosine(a: dict[str, int], b: dict[str, int]) -> float:
    if not a or not b:
        return 0.0
    shared = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in shared)
    na = sqrt(sum(v * v for v in a.values()))
    nb = sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class Document:
    collection: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    vector: dict[str, int] = field(default_factory=dict)


class LocalVectorStore:
    def __init__(self) -> None:
        self._docs: list[Document] = []
        self._seed()

    def _seed(self) -> None:
        for doc in STYLE_GUIDES:
            self.add("style_guides", doc["id"], doc["text"], {"type": "style"})
        for collection, docs in REMOTION_DOCS.items():
            for doc in docs:
                self.add(collection, doc["id"], doc["text"], {"type": "remotion"})

    def add(self, collection: str, doc_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        self._docs.append(
            Document(
                collection=collection,
                doc_id=doc_id,
                text=text,
                metadata=metadata or {},
                vector=_vectorize(text),
            )
        )

    def query(self, collection: str, text: str, top_k: int = 3) -> list[dict[str, Any]]:
        qv = _vectorize(text)
        scored: list[tuple[float, Document]] = []
        for doc in self._docs:
            if doc.collection != collection:
                continue
            scored.append((_cosine(qv, doc.vector), doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "collection": doc.collection,
                "doc_id": doc.doc_id,
                "text": doc.text,
                "metadata": doc.metadata,
                "score": score,
            }
            for score, doc in scored[:top_k]
        ]


class RetrievalService:
    def __init__(self, store: LocalVectorStore | None = None) -> None:
        self.store = store or LocalVectorStore()

    def retrieve_style(self, intent_text: str, top_k: int) -> list[dict[str, Any]]:
        return self.store.query("style_guides", intent_text, top_k=top_k)

    def retrieve_remotion(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        docs = []
        for collection in ("remotion_components", "remotion_animation", "remotion_transition", "remotion_cli"):
            docs.extend(self.store.query(collection, query_text, top_k=top_k))
        docs.sort(key=lambda item: item["score"], reverse=True)
        return docs[:top_k]

