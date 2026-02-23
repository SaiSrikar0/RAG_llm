"""ChromaDB vector store integration."""

from __future__ import annotations

from uuid import uuid4

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings

from src.config import settings


class VectorStore:
    """Wrapper around Chroma collection operations."""

    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(
            path=settings.chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection: Collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset_collection(self) -> None:
        """Drop and recreate the configured collection."""
        try:
            self.client.delete_collection(settings.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[str], embeddings: list[list[float]], source: str) -> None:
        """Insert chunk texts with precomputed embeddings into Chroma."""
        if not chunks:
            return
        ids = [str(uuid4()) for _ in chunks]
        metadata = [{"source": source, "chunk_index": idx} for idx in range(len(chunks))]
        self.collection.add(
            ids=ids,
            documents=chunks,
            metadatas=metadata,
            embeddings=embeddings,
        )

    def query(self, query_embedding: list[float], top_k: int) -> dict:
        """Query collection and return Chroma raw results."""
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

    def count(self) -> int:
        return self.collection.count()
