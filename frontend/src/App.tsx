import { type ChangeEvent, type FormEvent, useEffect, useState } from 'react'
import './App.css'

type DocumentInfo = {
  document_id: string
  document_name: string
  page_count: number
  file_size_bytes: number
  uploaded_at: string
}

type SearchResult = {
  document_id: string
  document_name: string
  page_number: number
  chunk_index: number
  text: string
  score: number
}

type AnswerResponse = {
  answer: string
  supported: boolean
  evidence: SearchResult[]
}

type UploadResponse = DocumentInfo & { duplicate: boolean }

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').trim().replace(/\/+$/, '')
const API_REQUEST_TIMEOUT_MS = 120_000

async function requestJson<T>(url: string, options?: RequestInit): Promise<T> {
  const apiPath = url.replace(/^\/api(?=\/|$)/, '')
  const requestUrl = API_BASE_URL ? `${API_BASE_URL}${apiPath}` : url
  let response: Response
  try {
    response = await fetch(requestUrl, {
      ...options,
      signal: options?.signal ?? AbortSignal.timeout(API_REQUEST_TIMEOUT_MS),
      headers: {
        ...(options?.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
        ...(options?.headers ?? {}),
      },
    })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'TimeoutError') {
      throw new Error('The request timed out after 120 seconds. The backend may be starting or processing the PDF; retry shortly.')
    }
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new Error('The request was cancelled before the backend responded.')
    }
    if (error instanceof TypeError) {
      throw new Error('The browser could not complete the backend request. Check your connection; if the backend is online, verify that this site origin is allowed by backend CORS.')
    }
    throw new Error(`The request could not be completed: ${error instanceof Error ? error.message : 'Unknown error.'}`)
  }

  if (!response.ok) {
    let detail = ''
    try {
      const payload: unknown = await response.clone().json()
      if (typeof payload === 'object' && payload !== null && 'detail' in payload) {
        const responseDetail = payload.detail
        detail = typeof responseDetail === 'string' ? responseDetail : JSON.stringify(responseDetail)
      }
    } catch {
      detail = (await response.text()).trim()
    }
    detail ||= response.statusText || 'No error details returned.'
    throw new Error(`Backend returned HTTP ${response.status}: ${detail}`)
  }

  return (await response.json()) as T
}

function App() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([])
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [chunkSize, setChunkSize] = useState(350)
  const [chunkOverlap, setChunkOverlap] = useState(60)
  const [query, setQuery] = useState('How does semantic search work?')
  const [askQuery, setAskQuery] = useState('How does semantic search work?')
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [answer, setAnswer] = useState('')
  const [answerEvidence, setAnswerEvidence] = useState<SearchResult[]>([])
  const [errorMessage, setErrorMessage] = useState('')
  const [noticeMessage, setNoticeMessage] = useState('')
  const [isLoadingDocuments, setIsLoadingDocuments] = useState(true)
  const [isUploading, setIsUploading] = useState(false)
  const [isSearching, setIsSearching] = useState(false)
  const [isAsking, setIsAsking] = useState(false)
  const [deletingDocumentId, setDeletingDocumentId] = useState<string | null>(null)

  const loadDocuments = async () => {
    setIsLoadingDocuments(true)
    try {
      const response = await requestJson<{ documents: DocumentInfo[] }>('/api/documents')
      setDocuments(response.documents ?? [])
    } finally {
      setIsLoadingDocuments(false)
    }
  }

  useEffect(() => {
    let isMounted = true
    requestJson<{ documents: DocumentInfo[] }>('/api/documents')
      .then((response) => {
        if (isMounted) {
          setDocuments(response.documents ?? [])
        }
      })
      .catch((error: unknown) => {
        if (isMounted) {
          setErrorMessage(error instanceof Error ? `Could not load documents: ${error.message}` : 'Could not load documents.')
        }
      })
      .finally(() => {
        if (isMounted) {
          setIsLoadingDocuments(false)
        }
      })
    return () => {
      isMounted = false
    }
  }, [])

  const onFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null
    setSelectedFile(file)
  }

  const handleUpload = async (event: FormEvent) => {
    event.preventDefault()
    if (!selectedFile) {
      setErrorMessage('Please choose a PDF file to upload.')
      return
    }
    setErrorMessage('')
    setNoticeMessage('')
    setIsUploading(true)

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)
      formData.append('chunk_size', String(chunkSize))
      formData.append('chunk_overlap', String(chunkOverlap))

      const response = await requestJson<UploadResponse>('/api/documents/upload', {
        method: 'POST',
        body: formData,
      })

      setSelectedFile(null)
      const fileInput = document.getElementById('pdf-file') as HTMLInputElement | null
      if (fileInput) {
        fileInput.value = ''
      }
      try {
        await loadDocuments()
      } catch (error) {
        setNoticeMessage(
          response.duplicate
            ? `${response.document_name} is already in the library.`
            : `${response.document_name} was uploaded and indexed.`,
        )
        setErrorMessage(
          `Upload succeeded, but the document list could not be refreshed: ${error instanceof Error ? error.message : 'Unknown error.'}`,
        )
        return
      }
      setNoticeMessage(
        response.duplicate
          ? `${response.document_name} is already in the library.`
          : `${response.document_name} was uploaded and indexed.`,
      )
    } catch (error) {
      setErrorMessage(error instanceof Error ? `Upload failed: ${error.message}` : 'Upload failed.')
    } finally {
      setIsUploading(false)
    }
  }

  const handleDelete = async (documentId: string) => {
    setErrorMessage('')
    setNoticeMessage('')
    setDeletingDocumentId(documentId)
    try {
      await requestJson<{ deleted: boolean }>(`/api/documents/${documentId}`, {
        method: 'DELETE',
      })
      setDocuments((currentDocuments) =>
        currentDocuments.filter((document) => document.document_id !== documentId),
      )
      setSearchResults((results) => results.filter((result) => result.document_id !== documentId))
      setAnswer('')
      setAnswerEvidence([])
      setNoticeMessage('Document removed. Search again to refresh passages.')
    } catch (error) {
      setErrorMessage(error instanceof Error ? `Delete failed: ${error.message}` : 'Delete failed.')
    } finally {
      setDeletingDocumentId(null)
    }
  }

  const handleSearch = async () => {
    if (!query.trim()) {
      setErrorMessage('Enter a search query first.')
      return
    }
    setIsSearching(true)
    setErrorMessage('')
    setNoticeMessage('')
    setSearchResults([])

    try {
      const response = await requestJson<{ results: SearchResult[] }>('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, top_k: 5 }),
      })
      setSearchResults(response.results ?? [])
    } catch (error) {
      setErrorMessage(error instanceof Error ? `Search failed: ${error.message}` : 'Semantic search failed.')
    } finally {
      setIsSearching(false)
    }
  }

  const handleAsk = async () => {
    if (!askQuery.trim()) {
      setErrorMessage('Enter a question to ask the document analyst.')
      return
    }
    setIsAsking(true)
    setErrorMessage('')
    setNoticeMessage('')
    setAnswer('')
    setAnswerEvidence([])

    try {
      const response = await requestJson<AnswerResponse>('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: askQuery, top_k: 3 }),
      })
      setAnswer(response.answer)
      setAnswerEvidence(response.evidence ?? [])
    } catch (error) {
      setErrorMessage(error instanceof Error ? `Question failed: ${error.message}` : 'Question answering failed.')
    } finally {
      setIsAsking(false)
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Evidence-based research analyzer</p>
          <h1>DocuMind AI</h1>
        </div>
        <div className="status-pill">Semantic retrieval ready</div>
      </header>

      <section className="panel upload-panel">
        <div className="panel-header">
          <h2>Upload PDF</h2>
        </div>

        <form onSubmit={handleUpload} className="upload-form">
          <label className="file-field">
            <span>PDF document</span>
            <input id="pdf-file" type="file" accept="application/pdf" onChange={onFileChange} />
          </label>

          <div className="config-grid">
            <label>
              <span>Chunk size</span>
              <input
                type="number"
                min={50}
                max={1000}
                value={chunkSize}
                onChange={(event) => setChunkSize(Number(event.target.value))}
              />
            </label>
            <label>
              <span>Overlap</span>
              <input
                type="number"
                min={0}
                max={400}
                value={chunkOverlap}
                onChange={(event) => setChunkOverlap(Number(event.target.value))}
              />
            </label>
          </div>

          <button type="submit" className="primary-button" disabled={isUploading || !selectedFile}>
            {isUploading ? 'Uploading...' : 'Upload and index'}
          </button>
        </form>
      </section>

      <div className="content-grid">
        <section className="panel">
          <div className="panel-header">
            <h2>Document library</h2>
          </div>

          <div className="document-list">
            {isLoadingDocuments ? (
              <p className="loading-state">Loading document library...</p>
            ) : documents.length === 0 ? (
              <p className="empty-state">No PDFs uploaded yet.</p>
            ) : (
              documents.map((document) => (
                <article key={document.document_id} className="document-item">
                  <div>
                    <h3>{document.document_name}</h3>
                    <p>{document.page_count} pages</p>
                  </div>
                  <button
                    type="button"
                    onClick={() => handleDelete(document.document_id)}
                    disabled={deletingDocumentId !== null}
                  >
                    {deletingDocumentId === document.document_id ? 'Removing...' : 'Remove'}
                  </button>
                </article>
              ))
            )}
          </div>
        </section>

        <section className="panel">
          <div className="panel-header">
            <h2>Semantic search</h2>
          </div>

          <div className="search-box">
            <textarea
              rows={4}
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search by meaning, not exact keywords..."
            />
            <button type="button" className="primary-button" onClick={handleSearch} disabled={isSearching}>
              {isSearching ? 'Searching...' : 'Search evidence'}
            </button>
          </div>

          <div className="results-list">
            {isSearching ? (
              <p className="loading-state">Searching indexed passages...</p>
            ) : searchResults.length === 0 ? (
              <p className="empty-state">No matching passages yet.</p>
            ) : (
              searchResults.map((result) => (
                <article key={`${result.document_id}-${result.chunk_index}`} className="result-card">
                  <div className="result-meta">
                    <span>{result.document_name}</span>
                    <span>Page {result.page_number}</span>
                    <span>Score {result.score.toFixed(3)}</span>
                  </div>
                  <p>{result.text}</p>
                </article>
              ))
            )}
          </div>
        </section>
      </div>

      <section className="panel answer-panel">
        <div className="panel-header">
          <h2>Research question</h2>
        </div>

        <div className="question-box">
          <textarea
            rows={3}
            value={askQuery}
            onChange={(event) => setAskQuery(event.target.value)}
            placeholder="Ask a research question about the uploaded PDFs..."
          />
          <button type="button" className="primary-button" onClick={handleAsk} disabled={isAsking}>
            {isAsking ? 'Analyzing...' : 'Ask the documents'}
          </button>
        </div>

        {answer ? (
          <div className="answer-box">
            <h3>Answer</h3>
            <p>{answer}</p>
            <div className="evidence-panel">
              <h4>Evidence</h4>
              {answerEvidence.length === 0 ? (
                <p className="empty-state">No evidence was found for this question.</p>
              ) : (
                answerEvidence.map((evidence, index) => (
                  <div key={`${evidence.document_id}-${evidence.chunk_index}-${index}`} className="evidence-item">
                    <strong>{evidence.document_name}</strong>
                    <span>Page {evidence.page_number}</span>
                    <p>{evidence.text}</p>
                  </div>
                ))
              )}
            </div>
          </div>
        ) : null}
        {isAsking ? <p className="loading-state">Reviewing retrieved evidence...</p> : null}
      </section>

      {errorMessage ? <p className="error-banner">{errorMessage}</p> : null}
      {noticeMessage ? <p className="notice-banner">{noticeMessage}</p> : null}
    </main>
  )
}

export default App
