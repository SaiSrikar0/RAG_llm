"""Embedding adapters supporting OpenAI and SentenceTransformers."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from src.config import settings


class EmbeddingService:
    """Unified embedding service abstraction for query and document vectors."""

    def __init__(self) -> None:
        provider = settings.embedding_provider.lower().strip()
        self.provider = provider
        self._st_model: SentenceTransformer | None = None
        self._openai_client: OpenAI | None = None

        if provider == "openai":
            if not settings.openai_api_key:
                msg = "OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai"
                raise ValueError(msg)
            self._openai_client = OpenAI(api_key=settings.openai_api_key)
        else:
            self.provider = "sentence_transformers"
            self._st_model = SentenceTransformer(settings.embedding_model)

    @staticmethod
    def _normalize(vectors: Sequence[Sequence[float]]) -> list[list[float]]:
        arr = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (arr / norms).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.provider == "openai" and self._openai_client:
            result = self._openai_client.embeddings.create(
                model=settings.openai_embedding_model,
                input=texts,
            )
            vectors = [row.embedding for row in result.data]
            return self._normalize(vectors)

        assert self._st_model is not None
        vectors = self._st_model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        vectors = self.embed_documents([query])
        return vectors[0]
