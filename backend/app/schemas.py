from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class PageText(BaseModel):
    page_number: int = Field(..., ge=1)
    text: str


class Chunk(BaseModel):
    document_id: str
    document_name: str
    page_number: int = Field(..., ge=1)
    chunk_index: int = Field(..., ge=0)
    text: str
    embedding: Optional[list[float]] = Field(default=None, exclude=True)

    model_config = ConfigDict(from_attributes=True)


class UploadedDocument(BaseModel):
    document_id: str
    document_name: str
    page_count: int = Field(..., ge=1)
    file_size_bytes: int = Field(..., ge=1)
    uploaded_at: datetime


class UploadResponse(UploadedDocument):
    duplicate: bool = False


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)


class SearchResult(BaseModel):
    document_id: str
    document_name: str
    page_number: int
    chunk_index: int
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total_results: int


class AskRequest(SearchRequest):
    pass


class AskResponse(BaseModel):
    answer: str
    supported: bool
    evidence: list[SearchResult]


class DocumentListResponse(BaseModel):
    documents: list[UploadedDocument]


class DeleteResponse(BaseModel):
    document_id: str
    deleted: bool
