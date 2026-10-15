# DocuMind AI

DocuMind AI is an evidence-based research document analyzer built as a beginner-friendly MVP for PDF research workflows. It combines a Python FastAPI backend with a React + Vite dashboard so users can upload research PDFs, search by meaning, and inspect supporting passages with page citations.

## Project goals

- Extract text from research PDFs with PyMuPDF
- Generate overlapping text chunks with page metadata
- Use a pretrained Sentence Transformers model for semantic embeddings
- Rank relevant passages using cosine similarity
- Provide retrieval-based answers grounded in uploaded evidence
- Let users upload, review, and remove documents from a clean dashboard

## Architecture

- Backend: FastAPI, Pydantic validation, PyMuPDF, Sentence Transformers
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- Search method: cosine similarity over cached chunk embeddings, with identical passage deduplication
- Frontend: React + TypeScript + Vite
- Storage: in-memory document store for the MVP; identical extracted PDF page content is indexed once

## Folder structure

- backend/app/main.py — FastAPI application and routes
- backend/app/pdf_processor.py — PDF text extraction logic
- backend/app/chunker.py — overlapping chunk generation
- backend/app/semantic_search.py — embedding and cosine similarity utilities
- backend/app/schemas.py — Pydantic data models
- backend/tests/test_app.py — backend smoke tests
- backend/tests/evaluate_retrieval.py — basic Recall@K evaluation script
- frontend/ — React dashboard
- sample_research.pdf — sample research PDF included for demos
- requirements.txt — Python dependencies
- .env.example — environment variable template

## Installation

1. Open PowerShell in the project root.
2. Create or reuse the workspace virtual environment.
3. Install Python dependencies:

   .\.venv\Scripts\python.exe -m pip install -r requirements.txt

4. Install frontend dependencies:

   cd frontend
   npm install

## Running the app

### Backend

From the project root:

   .\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8001

### Frontend

From the frontend folder:

   npm run dev -- --host 0.0.0.0

The Vite dev server proxies requests to the FastAPI backend at port 8001.

See [DEPLOYMENT.md](./DEPLOYMENT.md) for Render deployment settings, environment variables, startup behavior, and the current memory limitation on the Free backend instance.

## API overview

- GET /health — checks service status
- POST /documents/upload — uploads a PDF and stores indexed chunks; identical content reuses the existing document and returns `duplicate: true`
- GET /documents — lists uploaded documents
- DELETE /documents/{document_id} — removes a document
- POST /search — finds relevant passages by semantic similarity
- POST /ask — returns a retrieval-based answer with evidence references

FastAPI automatically generates interactive API docs at:

- http://127.0.0.1:8001/docs
- http://127.0.0.1:8001/redoc

## How semantic search works

The system stores each document as overlapping chunks, then converts each chunk and the user query into embedding vectors using a pretrained language model. The embeddings are compared with cosine similarity, which measures how aligned two vectors are in meaning space. Higher scores indicate stronger semantic relevance.

This is different from keyword matching because the model can match based on meaning rather than exact text overlap.

## Evidence policy

Question answering is extractive, not LLM-generated: it ranks sentences from relevant retrieved passages against the user's question, returns up to three distinct matching sentences, and cites each source page. Sentences from the same page share one citation. Both passage and sentence similarity must meet relevance cutoffs before text is used as evidence. If no sentence is relevant enough, the response explains that the documents do not contain enough information. Search results likewise contain distinct passages while preserving the best-ranked filename, page number, and similarity score.

Uploads are considered duplicates when their extracted, page-numbered text matches, even if the uploaded filename or PDF metadata differs. Removing a document removes its chunks and cached embeddings from the in-memory search corpus as well as its library entry.

## Demo and evaluation

A sample research PDF is included for local testing. You can also assess retrieval quality with:

   .\.venv\Scripts\python.exe backend/tests/evaluate_retrieval.py

This script reports a simple Recall@K-style result based on the sample document and sample query set. The numbers are measured in the local environment and should be treated as a practical diagnostic metric, not a published benchmark.

## Troubleshooting

### PDF upload fails

- Ensure the file is a real PDF
- Check that the file is not empty or corrupted
- Keep uploads below 15 MB for the default MVP limit
- If the PDF is scanned or image-only, the app will return a clear error message

### Model download stalls or fails

- The app loads sentence-transformers/all-MiniLM-L6-v2 on first use
- This may require internet access and a small amount of disk space
- If the download is blocked, the app will fail honestly instead of hiding the issue

### Frontend cannot reach backend

- Make sure the backend is running on the configured port (currently 8001 in this environment)
- Confirm the Vite proxy is configured in frontend/vite.config.ts
- Check the browser console for network errors

## Limitations

- The MVP keeps documents in an in-memory store, which resets on restart
- Uploaded documents may disappear after a process restart, redeploy, or instance replacement; there is no persistent document storage
- The answering mode is retrieval-based rather than generative AI
- It does not yet support OCR for scanned PDFs or database persistence
- Model quality depends on the quality and relevance of uploaded PDFs

## Beginner-friendly notes

- The app is intentionally simple and readable so the logic is easy to study
- Chunking, embeddings, and retrieval are separated into small modules
- Pydantic models and explicit validation help prevent malformed inputs

## Next possible improvements

- SQLite persistence for uploaded documents
- Better OCR for scanned PDFs
- Optional LLM integration with environment-based API keys only
- Better ranking explanations and evaluation dashboards
