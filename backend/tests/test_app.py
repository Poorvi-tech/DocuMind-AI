from io import BytesIO
import hashlib
from pathlib import Path
import re

import fitz
import numpy as np
import pytest
from fastapi.testclient import TestClient
from fastapi.middleware.cors import CORSMiddleware

from backend.app import main
from backend.app.main import app, document_store, search_engine
from backend.app.chunker import chunk_text
from backend.app.pdf_processor import extract_pdf_text
from backend.app.semantic_search import SemanticSearchEngine


client = TestClient(app)
TEST_STOP_WORDS = {"a", "and", "by", "does", "how", "is", "it", "of", "the", "what"}


def _make_pdf_bytes(text: str) -> bytes:
    return _make_multi_page_pdf_bytes([text])


def _make_multi_page_pdf_bytes(texts: list[str]) -> bytes:
    pdf_buffer = BytesIO()
    document = fitz.open()
    for text in texts:
        page = document.new_page()
        page.insert_text((72, 72), text)
    document.save(pdf_buffer)
    document.close()
    return pdf_buffer.getvalue()


def _upload_sample_pdf_with_model_embeddings(monkeypatch):
    monkeypatch.setattr(
        search_engine,
        "embed",
        lambda texts: search_engine.__class__.embed(search_engine, texts),
    )
    sample_path = Path(__file__).resolve().parents[2] / "sample_research.pdf"
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                sample_path.name,
                sample_path.read_bytes(),
                "application/pdf",
            )
        },
    )
    assert response.status_code == 200


@pytest.fixture(autouse=True)
def isolate_document_store(monkeypatch):
    existing_documents = dict(document_store)
    document_store.clear()
    monkeypatch.setattr(main, "_sample_document_seeded", True)

    def embed(texts: list[str]) -> np.ndarray:
        embeddings = np.zeros((len(texts), 4096), dtype=np.float32)
        for row, text in zip(embeddings, texts):
            for token in set(re.findall(r"\w+", text.casefold())):
                if token in TEST_STOP_WORDS:
                    continue
                index = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % len(row)
                row[index] += 1
        return embeddings

    monkeypatch.setattr(search_engine, "embed", embed)
    yield
    document_store.clear()
    document_store.update(existing_documents)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_semantic_search_engine_loads_model_lazily():
    engine = SemanticSearchEngine()

    assert engine.is_loaded is False


def test_sample_document_is_seeded_once_on_demand(monkeypatch):
    monkeypatch.setattr(main, "_sample_document_seeded", False)

    main._seed_default_documents()
    first_document_count = len(document_store)
    main._seed_default_documents()

    assert first_document_count == 1
    assert len(document_store) == 1
    sample = next(iter(document_store.values()))
    assert sample["document"].document_name == "sample_research.pdf"
    assert all(chunk.embedding is not None for chunk in sample["chunks"])


def test_cors_defaults_to_explicit_origins_without_credentials():
    cors_middleware = next(
        middleware for middleware in app.user_middleware
        if middleware.cls is CORSMiddleware
    )

    assert cors_middleware.kwargs["allow_origins"] == [
        "https://docu-mind-ai-phi.vercel.app",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    assert cors_middleware.kwargs["allow_credentials"] is False


def test_cors_env_origins_extend_required_and_local_origins():
    origins = main._parse_frontend_origins(
        "https://preview.example.com, http://localhost:5173/"
    )

    assert origins == [
        "https://docu-mind-ai-phi.vercel.app",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://preview.example.com",
    ]


def test_cors_env_rejects_wildcard_origin():
    with pytest.raises(ValueError, match="explicit origins"):
        main._parse_frontend_origins("*")


@pytest.mark.parametrize(
    ("path", "method"),
    [
        ("/documents", "GET"),
        ("/documents/upload", "POST"),
    ],
)
def test_cors_preflight_allows_deployed_frontend_origin(path, method):
    response = client.options(
        path,
        headers={
            "Origin": "https://docu-mind-ai-phi.vercel.app",
            "Access-Control-Request-Method": method,
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == (
        "https://docu-mind-ai-phi.vercel.app"
    )
    assert method in response.headers["access-control-allow-methods"]
    assert "content-type" in response.headers["access-control-allow-headers"]


def test_upload_rejects_content_over_configured_limit(monkeypatch):
    monkeypatch.setattr(main, "MAX_FILE_SIZE_BYTES", 8)
    monkeypatch.setattr(main, "MAX_FILE_SIZE_MB", 1)

    response = client.post(
        "/documents/upload",
        files={"file": ("large.pdf", b"x" * 9, "application/pdf")},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "File exceeds 1 MB limit."


def test_extract_pdf_text_returns_pages():
    text = "Artificial Intelligence helps machines reason.\n\nNatural Language Processing analyzes language."
    pages = extract_pdf_text(_make_pdf_bytes(text), "sample.pdf")
    assert len(pages) == 1
    assert "Artificial Intelligence" in pages[0].text
    assert pages[0].page_number == 1


def test_chunking_preserves_page_metadata():
    long_text = (
        "Artificial Intelligence enables computers to perform tasks that normally require human intelligence. "
        "Natural Language Processing helps systems interpret language and understand meaning. "
        "Semantic search identifies relevant passages based on meaning rather than exact keyword matches. "
        "This retrieval approach is useful for research documents because it surfaces supporting evidence. "
        "Evidence ranking uses similarity scores to prioritize the most relevant passages for a user query."
    )
    chunks = chunk_text(
        long_text,
        chunk_size=35,
        chunk_overlap=10,
        page_number=2,
        document_name="sample.pdf",
    )
    assert len(chunks) > 1
    assert all(chunk.document_name == "sample.pdf" for chunk in chunks)
    assert all(chunk.page_number == 2 for chunk in chunks)
    assert any("Artificial Intelligence" in chunk.text for chunk in chunks)


def test_document_upload_and_delete_flow():
    pdf_bytes = _make_pdf_bytes("Semantic search retrieves relevant evidence based on meaning.")
    response = client.post(
        "/documents/upload",
        files={"file": ("research.pdf", pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["duplicate"] is False
    assert payload["document_name"] == "research.pdf"
    assert payload["page_count"] == 1

    list_response = client.get("/documents")
    assert list_response.status_code == 200
    assert len(list_response.json()["documents"]) == 1

    delete_response = client.delete(f"/documents/{payload['document_id']}")
    assert delete_response.status_code == 200
    assert delete_response.json()["deleted"] is True


def test_duplicate_upload_reuses_existing_document():
    pdf_bytes = _make_pdf_bytes("Semantic search retrieves relevant evidence.")
    first_response = client.post(
        "/documents/upload",
        files={"file": ("research.pdf", pdf_bytes, "application/pdf")},
    )
    second_response = client.post(
        "/documents/upload",
        files={"file": ("renamed-copy.pdf", pdf_bytes, "application/pdf")},
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    first_document = first_response.json()
    duplicate_response = second_response.json()
    assert duplicate_response["duplicate"] is True
    assert duplicate_response["document_id"] == first_document["document_id"]
    assert len(client.get("/documents").json()["documents"]) == 1


def test_search_returns_distinct_passages_with_metadata():
    pdf_bytes = _make_multi_page_pdf_bytes(
        ["Semantic search retrieves relevant passages by meaning."] * 2
    )
    upload_response = client.post(
        "/documents/upload",
        files={"file": ("passages.pdf", pdf_bytes, "application/pdf")},
    )
    document_id = upload_response.json()["document_id"]

    response = client.post(
        "/search",
        json={"query": "How does semantic search retrieve passages?", "top_k": 5},
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["document_id"] == document_id
    assert results[0]["document_name"] == "passages.pdf"
    assert results[0]["page_number"] in (1, 2)
    assert results[0]["score"] > 0
    assert results[0]["text"] == "Semantic search retrieves relevant passages by meaning."


def test_deleting_document_removes_its_search_results():
    upload_response = client.post(
        "/documents/upload",
        files={
            "file": (
                "temporary.pdf",
                _make_pdf_bytes("Semantic search retrieves relevant passages."),
                "application/pdf",
            )
        },
    )
    document_id = upload_response.json()["document_id"]
    query = {"query": "How does semantic search retrieve passages?", "top_k": 5}
    assert client.post("/search", json=query).json()["results"]

    delete_response = client.delete(f"/documents/{document_id}")

    assert delete_response.status_code == 200
    assert client.get("/documents").json()["documents"] == []
    assert client.post("/search", json=query).json()["results"] == []
    ask_response = client.post(
        "/ask",
        json={"query": query["query"], "top_k": query["top_k"]},
    )
    assert ask_response.json()["supported"] is False
    assert ask_response.json()["evidence"] == []


def test_answer_cites_retrieved_evidence():
    client.post(
        "/documents/upload",
        files={
            "file": (
                "citations.pdf",
                _make_pdf_bytes(
                    "Semantic search identifies relevant passages by meaning. "
                    "It ranks evidence using similarity scores."
                ),
                "application/pdf",
            )
        },
    )

    response = client.post(
        "/ask",
        json={"query": "How does semantic search identify passages?", "top_k": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["supported"] is True
    assert "Semantic search identifies relevant passages by meaning." in payload["answer"]
    assert "[citations.pdf, p. 1]" in payload["answer"]
    assert payload["evidence"][0]["document_name"] == "citations.pdf"
    assert payload["evidence"][0]["page_number"] == 1


def test_answer_reports_insufficient_evidence():
    client.post(
        "/documents/upload",
        files={
            "file": (
                "coastal-report.pdf",
                _make_pdf_bytes("The report discusses coastal water sampling methods."),
                "application/pdf",
            )
        },
    )

    response = client.post(
        "/ask",
        json={"query": "What is the population of Mars?", "top_k": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["supported"] is False
    assert "not contain enough information" in payload["answer"]
    assert payload["evidence"] == []


def test_answer_selects_natural_language_processing_definition(monkeypatch):
    _upload_sample_pdf_with_model_embeddings(monkeypatch)

    response = client.post(
        "/ask",
        json={"query": "What is Natural Language Processing?", "top_k": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["supported"] is True
    assert "Natural Language Processing helps computers process and analyze human language." in payload["answer"]
    assert "Semantic search identifies relevant information" not in payload["answer"]
    assert payload["answer"].count("[sample_research.pdf, p. 1]") == 1
    assert len(payload["evidence"]) == 1


def test_answer_selects_how_semantic_search_works(monkeypatch):
    _upload_sample_pdf_with_model_embeddings(monkeypatch)

    response = client.post(
        "/ask",
        json={"query": "How does semantic search work?", "top_k": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["supported"] is True
    assert "Semantic search identifies relevant information based on meaning rather than exact keyword matches." in payload["answer"]
    assert "Natural Language Processing helps computers" not in payload["answer"]
    assert payload["answer"].count("[sample_research.pdf, p. 1]") == 1


def test_answer_rejects_question_absent_from_sample_pdf(monkeypatch):
    _upload_sample_pdf_with_model_embeddings(monkeypatch)

    response = client.post(
        "/ask",
        json={"query": "What is the population of Mars?", "top_k": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["supported"] is False
    assert "not contain enough information" in payload["answer"]
    assert payload["evidence"] == []


def test_search_returns_ranked_evidence():
    client.post(
        "/documents/upload",
        files={
            "file": (
                "semantic-search.pdf",
                _make_pdf_bytes("Semantic search identifies relevant passages by meaning."),
                "application/pdf",
            )
        },
    )

    response = client.post(
        "/search",
        json={"query": "How does semantic search work?", "top_k": 3},
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert isinstance(results, list)
    assert len(results) == 1
    assert "score" in results[0]
    assert "document_name" in results[0]
