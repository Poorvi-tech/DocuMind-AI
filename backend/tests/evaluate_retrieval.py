from __future__ import annotations

from pathlib import Path

import numpy as np

from backend.app.chunker import chunk_text
from backend.app.pdf_processor import extract_pdf_text
from backend.app.semantic_search import SemanticSearchEngine

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PDF = ROOT / "sample_research.pdf"


def _load_sample_chunks():
    pages = extract_pdf_text(SAMPLE_PDF.read_bytes(), SAMPLE_PDF.name)
    chunks = []
    for page in pages:
        chunks.extend(
            chunk_text(
                page.text,
                chunk_size=180,
                chunk_overlap=40,
                page_number=page.page_number,
                document_name=SAMPLE_PDF.name,
                document_id="sample",
            )
        )
    return chunks


def evaluate_recall_at_k():
    engine = SemanticSearchEngine()
    chunks = _load_sample_chunks()
    embeddings = engine.embed([chunk.text for chunk in chunks])
    queries = {
        "artificial intelligence": [0],
        "natural language processing": [1],
        "semantic search retrieves meaning": [2],
    }

    results = []
    for query, relevant_indices in queries.items():
        query_embedding = engine.embed([query])[0]
        scores = engine.cosine_similarity(query_embedding, embeddings)
        ranked = np.argsort(scores)[::-1]
        retrieved = set(int(index) for index in ranked[:3])
        relevant = set(relevant_indices)
        recall = len(retrieved & relevant) / len(relevant) if relevant else 0.0
        results.append({"query": query, "recall@3": recall})

    return results


if __name__ == "__main__":
    print("Running Recall@K evaluation on the sample research document.")
    for row in evaluate_recall_at_k():
        print(row)
