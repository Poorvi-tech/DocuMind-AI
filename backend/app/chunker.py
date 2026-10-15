from __future__ import annotations

import re
from typing import List

from backend.app.schemas import Chunk


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(
    text: str,
    chunk_size: int = 350,
    chunk_overlap: int = 60,
    page_number: int = 1,
    document_name: str = "document.pdf",
    document_id: str = "",
) -> List[Chunk]:
    normalized_text = normalize_whitespace(text)
    if not normalized_text:
        return []

    chunk_size = max(10, int(chunk_size))
    chunk_overlap = max(0, min(int(chunk_overlap), chunk_size - 1))
    words = normalized_text.split()
    if len(words) <= chunk_size:
        return [
            Chunk(
                document_id=document_id,
                document_name=document_name,
                page_number=page_number,
                chunk_index=0,
                text=normalized_text,
            )
        ]

    step = max(1, chunk_size - chunk_overlap)
    chunks: List[Chunk] = []
    for index, start in enumerate(range(0, len(words), step)):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_value = " ".join(chunk_words).strip()
        if not chunk_value:
            continue
        chunks.append(
            Chunk(
                document_id=document_id,
                document_name=document_name,
                page_number=page_number,
                chunk_index=index,
                text=chunk_value,
            )
        )
        if end >= len(words):
            break
    if not chunks:
        return [
            Chunk(
                document_id=document_id,
                document_name=document_name,
                page_number=page_number,
                chunk_index=0,
                text=normalized_text,
            )
        ]
    return chunks
