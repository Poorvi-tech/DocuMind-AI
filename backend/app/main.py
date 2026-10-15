from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.app.chunker import chunk_text
from backend.app.pdf_processor import extract_pdf_text
from backend.app.schemas import (
    AskRequest,
    AskResponse,
    Chunk,
    DeleteResponse,
    DocumentListResponse,
    PageText,
    SearchRequest,
    SearchResponse,
    SearchResult,
    UploadResponse,
    UploadedDocument,
)
from backend.app.semantic_search import SemanticSearchEngine

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "15"))
if MAX_FILE_SIZE_MB < 1:
    raise ValueError("MAX_FILE_SIZE_MB must be a positive integer.")
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
DEFAULT_CHUNK_SIZE = 350
DEFAULT_CHUNK_OVERLAP = 60
MIN_ANSWER_SCORE = 0.35
MIN_ANSWER_SENTENCE_SCORE = 0.5
INSUFFICIENT_EVIDENCE_ANSWER = (
    "The uploaded documents do not contain enough information to answer that question with evidence."
)

app = FastAPI(title="DocuMind AI", version="1.0.0")
frontend_origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
if "*" in frontend_origins:
    raise ValueError("FRONTEND_ORIGINS must contain explicit origins; '*' is not allowed.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

search_engine = SemanticSearchEngine()
document_store: dict[str, dict[str, Any]] = {}


def _all_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for item in document_store.values():
        chunks.extend(item["chunks"])
    return chunks


def _document_fingerprint(pages: list[PageText]) -> str:
    page_content = "\n".join(f"{page.page_number}:{page.text}" for page in pages)
    return hashlib.sha256(page_content.encode("utf-8")).hexdigest()


def _normalized_passage(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _build_search_results(query_text: str, top_k: int) -> list[SearchResult]:
    chunks = _all_chunks()
    if not chunks:
        return []

    indexed_chunks = [chunk for chunk in chunks if chunk.embedding is not None]
    if not indexed_chunks:
        return []

    chunk_embeddings = np.asarray(
        [chunk.embedding for chunk in indexed_chunks],
        dtype=np.float32,
    )
    if chunk_embeddings.size == 0:
        return []

    query_embedding = search_engine.embed([query_text])[0]
    scores = search_engine.cosine_similarity(query_embedding, chunk_embeddings)
    ranked_index = np.argsort(scores)[::-1]

    results: list[SearchResult] = []
    seen_passages: set[str] = set()
    for idx in ranked_index:
        chunk = indexed_chunks[int(idx)]
        passage_key = _normalized_passage(chunk.text)
        if passage_key in seen_passages:
            continue
        seen_passages.add(passage_key)
        score = float(scores[int(idx)])
        results.append(
            SearchResult(
                document_id=chunk.document_id,
                document_name=chunk.document_name,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                score=score,
            )
        )
        if len(results) == top_k:
            break
    return results


def _grounded_answer(query_text: str, evidence: list[SearchResult]) -> AskResponse:
    supporting_evidence = [
        item for item in evidence if item.score >= MIN_ANSWER_SCORE
    ]
    if not supporting_evidence:
        return AskResponse(
            answer=INSUFFICIENT_EVIDENCE_ANSWER,
            supported=False,
            evidence=[],
        )

    sentence_sources: list[tuple[str, SearchResult]] = []
    seen_sentences: set[str] = set()
    for item in supporting_evidence:
        sentences = re.split(r"(?<=[.!?])\s+", item.text.strip())
        for sentence in sentences:
            sentence = sentence.strip()
            sentence_key = _normalized_passage(sentence)
            if not sentence or sentence_key in seen_sentences:
                continue
            seen_sentences.add(sentence_key)
            sentence_sources.append((sentence, item))

    if not sentence_sources:
        return AskResponse(
            answer=INSUFFICIENT_EVIDENCE_ANSWER,
            supported=False,
            evidence=[],
        )

    sentence_embeddings = search_engine.embed(
        [sentence for sentence, _ in sentence_sources]
    )
    query_embedding = search_engine.embed([query_text])[0]
    sentence_scores = search_engine.cosine_similarity(
        query_embedding,
        sentence_embeddings,
    )
    ranked_sentences = sorted(
        (
            (float(sentence_scores[index]), sentence, source)
            for index, (sentence, source) in enumerate(sentence_sources)
            if sentence_scores[index] >= MIN_ANSWER_SENTENCE_SCORE
        ),
        key=lambda candidate: candidate[0],
        reverse=True,
    )[:3]
    if not ranked_sentences:
        return AskResponse(
            answer=INSUFFICIENT_EVIDENCE_ANSWER,
            supported=False,
            evidence=[],
        )

    grouped_sentences: dict[tuple[str, str, int], list[str]] = {}
    grouped_evidence: dict[tuple[str, str, int], SearchResult] = {}
    answer_length = 0
    for _, sentence, source in ranked_sentences:
        source_key = (source.document_id, source.document_name, source.page_number)
        citation = f"[{source.document_name}, p. {source.page_number}]"
        addition_length = len(sentence) + (1 if grouped_sentences.get(source_key) else 0)
        if answer_length + addition_length + len(citation) + 1 > 800:
            continue
        grouped_sentences.setdefault(source_key, []).append(sentence)
        grouped_evidence[source_key] = source
        answer_length += addition_length

    if not grouped_sentences:
        return AskResponse(
            answer=INSUFFICIENT_EVIDENCE_ANSWER,
            supported=False,
            evidence=[],
        )

    answer_parts = []
    for source_key, sentences in grouped_sentences.items():
        source = grouped_evidence[source_key]
        citation = f"[{source.document_name}, p. {source.page_number}]"
        answer_parts.append(f"{' '.join(sentences)} {citation}")

    return AskResponse(
        answer=" ".join(answer_parts),
        supported=True,
        evidence=list(grouped_evidence.values()),
    )


def _seed_default_documents() -> None:
    sample_path = Path(__file__).resolve().parents[2] / "sample_research.pdf"
    if not sample_path.exists():
        return

    if any(item["document"].document_name == sample_path.name for item in document_store.values()):
        return

    try:
        file_bytes = sample_path.read_bytes()
        pages = extract_pdf_text(file_bytes, sample_path.name)
    except ValueError:
        return

    fingerprint = _document_fingerprint(pages)
    if any(item.get("fingerprint") == fingerprint for item in document_store.values()):
        return

    document_id = str(uuid4())
    chunks: list[Chunk] = []
    for page in pages:
        page_chunks = chunk_text(
            page.text,
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
            page_number=page.page_number,
            document_name=sample_path.name,
            document_id=document_id,
        )
        chunks.extend(page_chunks)

    embeddings = search_engine.embed([chunk.text for chunk in chunks])
    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding.tolist()

    document_store[document_id] = {
        "document": UploadedDocument(
            document_id=document_id,
            document_name=sample_path.name,
            page_count=len(pages),
            file_size_bytes=len(file_bytes),
            uploaded_at=datetime.now(timezone.utc),
        ),
        "chunks": chunks,
        "fingerprint": fingerprint,
    }


_seed_default_documents()


@app.on_event("startup")
async def startup_event() -> None:
    _seed_default_documents()


@app.get("/health")
def health_check() -> dict[str, Any]:
    return {
        "status": "ok",
        "documents": len(document_store),
        "model": "sentence-transformers/all-MiniLM-L6-v2",
    }


@app.get("/documents", response_model=DocumentListResponse)
def list_documents() -> DocumentListResponse:
    documents = [item["document"] for item in document_store.values()]
    return DocumentListResponse(documents=documents)


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(DEFAULT_CHUNK_SIZE),
    chunk_overlap: int = Form(DEFAULT_CHUNK_OVERLAP),
) -> UploadResponse:
    if file.filename is None or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    file_bytes = await file.read(MAX_FILE_SIZE_BYTES + 1)
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="The uploaded PDF is empty.")
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {MAX_FILE_SIZE_MB} MB limit.",
        )

    try:
        pages = extract_pdf_text(file_bytes, file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    fingerprint = _document_fingerprint(pages)
    for item in document_store.values():
        if item.get("fingerprint") == fingerprint:
            return UploadResponse(**item["document"].model_dump(), duplicate=True)

    document_id = str(uuid4())
    chunks: list[Chunk] = []
    for page in pages:
        page_chunks = chunk_text(
            page.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            page_number=page.page_number,
            document_name=file.filename,
            document_id=document_id,
        )
        chunks.extend(page_chunks)

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No text chunks could be created from this PDF. Please upload a readable document.",
        )

    embeddings = search_engine.embed([chunk.text for chunk in chunks])
    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding.tolist()

    document = UploadedDocument(
        document_id=document_id,
        document_name=file.filename,
        page_count=len(pages),
        file_size_bytes=len(file_bytes),
        uploaded_at=datetime.now(timezone.utc),
    )
    document_store[document_id] = {"document": document, "chunks": chunks}
    document_store[document_id]["fingerprint"] = fingerprint
    return UploadResponse(**document.model_dump(), duplicate=False)


@app.delete("/documents/{document_id}", response_model=DeleteResponse)
def delete_document(document_id: str) -> DeleteResponse:
    if document_id not in document_store:
        raise HTTPException(status_code=404, detail="Document not found.")
    del document_store[document_id]
    return DeleteResponse(document_id=document_id, deleted=True)


@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest) -> SearchResponse:
    query_text = request.query.strip()
    if not document_store:
        return SearchResponse(query=query_text, results=[], total_results=0)

    results = _build_search_results(query_text, request.top_k)
    return SearchResponse(query=query_text, results=results, total_results=len(results))


@app.post("/ask", response_model=AskResponse)
async def answer_question(request: AskRequest) -> AskResponse:
    query_text = request.query.strip()
    evidence = _build_search_results(query_text, request.top_k)
    return _grounded_answer(query_text, evidence)
