from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import chromadb

from app.rag.seeds import REMOTION_DOCS, STYLE_GUIDES
from app.settings import SETTINGS
from app.services.embedding_service import HashingEmbeddingFunction
from app.utils import ensure_dir


@dataclass
class RetrievalService:
    persist_dir: Path | str = SETTINGS.chroma_path
    top_k_default: int = 3

    def __post_init__(self) -> None:
        self.persist_dir = ensure_dir(Path(self.persist_dir) / uuid4().hex[:10])
        self.embedding_function = HashingEmbeddingFunction()
        self.client = chromadb.Client()
        self.collections = {
            "style_guides": self.client.get_or_create_collection(name="style_guides", metadata={"hnsw:space": "cosine"}),
            "remotion_components": self.client.get_or_create_collection(name="remotion_components", metadata={"hnsw:space": "cosine"}),
            "remotion_animation": self.client.get_or_create_collection(name="remotion_animation", metadata={"hnsw:space": "cosine"}),
            "remotion_transition": self.client.get_or_create_collection(name="remotion_transition", metadata={"hnsw:space": "cosine"}),
            "remotion_cli": self.client.get_or_create_collection(name="remotion_cli", metadata={"hnsw:space": "cosine"}),
        }
        self._seed_if_needed()

    def _seed_if_needed(self) -> None:
        self._seed_collection("style_guides", STYLE_GUIDES)
        for collection, docs in REMOTION_DOCS.items():
            self._seed_collection(collection, docs)

    def _seed_collection(self, collection_name: str, docs: list[dict[str, str]]) -> None:
        collection = self.collections[collection_name]
        if collection.count() > 0:
            return
        collection.add(
            ids=[doc["id"] for doc in docs],
            documents=[doc["text"] for doc in docs],
            metadatas=[{"type": collection_name, "doc_id": doc["id"]} for doc in docs],
            embeddings=[self.embedding_function.embed_text(doc["text"]) for doc in docs],
        )

    def query(self, collection_name: str, text: str, top_k: int | None = None) -> list[dict[str, Any]]:
        collection = self.collections[collection_name]
        result = collection.query(
            query_embeddings=[self.embedding_function.embed_text(text)],
            n_results=top_k or self.top_k_default,
            include=["documents", "metadatas", "distances"],
        )
        return self._normalize_results(collection_name, result)

    def retrieve_style(self, intent_text: str, top_k: int) -> list[dict[str, Any]]:
        return self.query("style_guides", intent_text, top_k=top_k)

    def retrieve_remotion(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        for collection_name in ("remotion_components", "remotion_animation", "remotion_transition", "remotion_cli"):
            docs.extend(self.query(collection_name, query_text, top_k=top_k))
        docs.sort(key=lambda item: item["score"], reverse=True)
        return docs[:top_k]

    def _normalize_results(self, collection_name: str, result: dict[str, Any]) -> list[dict[str, Any]]:
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        normalized: list[dict[str, Any]] = []
        for index, document in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = float(distances[index]) if index < len(distances) else 1.0
            normalized.append(
                {
                    "collection": collection_name,
                    "doc_id": metadata.get("doc_id", ""),
                    "text": document,
                    "metadata": metadata,
                    "score": max(0.0, 1.0 - distance),
                }
            )
        return normalized
