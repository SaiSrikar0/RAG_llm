"""Text loading and preprocessing helpers for the RAG pipeline."""

from __future__ import annotations

import io
from typing import Iterable

import pypdf


def read_txt_bytes(file_bytes: bytes) -> str:
    """Decode plain text bytes into UTF-8 text."""
    return file_bytes.decode("utf-8", errors="ignore")


def read_pdf_bytes(file_bytes: bytes) -> str:
    """Extract text from each page in a PDF document."""
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """Chunk long text into overlapping windows for retrieval."""
    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunks.append(normalized[start:end])
        if end == len(normalized):
            break
        start = max(0, end - overlap)
    return chunks


def flatten(values: Iterable[Iterable[str]]) -> list[str]:
    """Flatten nested iterables into a single list."""
    return [item for sub in values for item in sub]
