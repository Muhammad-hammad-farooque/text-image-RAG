# Technical Specification — Text-Image RAG System

**Version:** 1.0.0
**Python:** 3.12
**Package Manager:** uv

---

## 1. Project Overview

An industry-grade multimodal Retrieval-Augmented Generation (RAG) system that ingests documents (PDF, DOCX, TXT, XLSX, CSV), extracts text, tables, and images with structure preserved, indexes them in a persistent vector store, and answers user questions through a streaming chat interface powered by a vision-capable LLM.

---

## 2. Goals

| Goal | Description |
|------|-------------|
| Multimodal retrieval | Retrieve relevant text chunks, tables, and images from a single unified pipeline |
| Table-aware parsing | Extract tables from PDFs and DOCX with structure preserved — never split mid-table |
| Multilingual support | Accept queries in 50+ languages without a translation step |
| High retrieval quality | HyDE + hybrid search (vector + BM25) + cross-encoder reranking |
| Production reliability | Async ingestion, persistent vector store, structured logging, tracing |
| CPU-only deployment | All embedding and reranking runs on CPU — no GPU required |
| Secure multi-user access | JWT authentication, per-user rate limiting, isolated document collections |

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          User (Browser)                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Chainlit UI    │  (port 8000)
                    └────────┬────────┘
                             │ HTTP
                    ┌────────▼────────┐
                    │   FastAPI App   │  (port 8001)
                    │  /ingest        │
                    │  /query         │
                    │  /health        │
                    └──┬──────────┬───┘
                       │          │
          ┌────────────▼──┐   ┌───▼────────────────┐
          │ Celery Worker │   │   RAG Pipeline     │
          │ (ingestion)   │   │   (query time)     │
          └────────┬──────┘   └───┬────────────────┘
                   │              │
        ┌──────────▼──┐    ┌──────▼──────┐    ┌──────────────┐
        │    Redis    │    │   Qdrant    │    │  Groq (LLM)  │
        │  (broker +  │    │  (vector    │    │  Llama 4     │
        │   cache)    │    │   store)    │    │  Scout       │
        └─────────────┘    └─────────────┘    └──────────────┘
```

---

## 4. Project File Structure

```
text_image_rag/
├── app.py                      # Chainlit frontend — UI and session management
├── api/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app — router registration, lifespan
│   ├── routes/
│   │   ├── ingest.py           # POST /ingest — document upload and processing
│   │   ├── query.py            # POST /query — RAG query endpoint
│   │   └── health.py           # GET /health — system health check
│   ├── schemas/
│   │   ├── ingest.py           # Pydantic models for ingest request/response
│   │   └── query.py            # Pydantic models for query request/response
│   └── dependencies.py         # Auth, rate limiter, shared FastAPI dependencies
├── rag/
│   ├── __init__.py
│   ├── parser/
│   │   ├── pdf.py              # PyMuPDF (text) + pdfplumber (tables) + image extraction
│   │   ├── docx.py             # python-docx text, table, and image extraction
│   │   ├── excel.py            # pandas Excel and CSV parsing
│   │   └── txt.py              # Plain text extraction
│   ├── embedder.py             # sentence-transformers multilingual embedding
│   ├── vector_store.py         # Qdrant client — upsert, search, collection management
│   ├── retriever.py            # Hybrid search (BM25 + vector) + RRF merging
│   ├── reranker.py             # cross-encoder reranking
│   ├── hyde.py                 # HyDE — hypothetical document generation + embedding
│   └── generator.py            # Prompt construction + Groq LLM streaming
├── tasks/
│   ├── __init__.py
│   └── ingest_task.py          # Celery task — async document ingestion pipeline
├── core/
│   ├── __init__.py
│   ├── config.py               # Pydantic Settings — all environment variables
│   ├── logger.py               # Loguru setup — JSON structured logging
│   └── security.py             # JWT creation and verification
├── tests/
│   ├── unit/
│   │   ├── test_parser.py      # Test PDF, DOCX, Excel parsing and table extraction
│   │   ├── test_embedder.py    # Test embedding shape and normalization
│   │   └── test_retriever.py   # Test BM25, vector search, RRF merging
│   └── integration/
│       └── test_pipeline.py    # End-to-end ingest → query → answer test
├── pyproject.toml              # uv project metadata and all dependencies
├── uv.lock                     # Locked dependency versions
├── requirements.txt            # pip fallback (mirrors pyproject.toml)
├── docker-compose.yml          # App + Qdrant + Redis + Celery worker
├── Dockerfile                  # Application container
├── .env.example                # Environment variable template
├── .python-version             # Python 3.12 (uv pin)
├── .gitignore
├── README.md
└── SPEC.md
```

---

## 5. Environment Variables

Managed via Pydantic Settings in `core/config.py`. All variables are validated at startup — the app will not start if a required variable is missing.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | Yes | — | Groq API key for LLM inference and HyDE generation |
| `LANGSMITH_API_KEY` | Yes | — | LangSmith key for end-to-end RAG tracing |
| `QDRANT_URL` | No | `http://localhost:6333` | Qdrant vector store host |
| `REDIS_URL` | No | `redis://localhost:6379` | Redis URL for Celery broker and cache |
| `JWT_SECRET_KEY` | Yes | — | Secret key for signing JWT tokens |
| `JWT_ALGORITHM` | No | `HS256` | JWT signing algorithm |
| `JWT_EXPIRE_MINUTES` | No | `60` | Token expiry in minutes |
| `EMBEDDING_MODEL` | No | `paraphrase-multilingual-MiniLM-L12-v2` | sentence-transformers model name |
| `RERANKER_MODEL` | No | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model name |
| `LLM_MODEL` | No | `meta-llama/llama-4-scout-17b-16e-instruct` | Groq model ID |
| `CHUNK_SIZE` | No | `500` | Text chunk size in characters |
| `CHUNK_OVERLAP` | No | `100` | Overlap between consecutive chunks |
| `RETRIEVAL_TOP_K` | No | `20` | Number of candidates before reranking |
| `RERANK_TOP_N` | No | `5` | Final top-N chunks after reranking |
| `MAX_FILE_SIZE_MB` | No | `50` | Maximum upload file size |
| `RATE_LIMIT` | No | `20/minute` | Per-user request rate limit |

---

## 6. API Specification

### 6.1 POST `/ingest`

Upload and process a document. Processing runs as a Celery background task.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | Document file (.pdf, .docx, .txt, .xlsx, .csv) |
| `collection_name` | string | No | Qdrant collection name (defaults to user ID) |

**Response** `202 Accepted`

```json
{
  "task_id": "c3d4e5f6-...",
  "status": "queued",
  "filename": "report.pdf",
  "message": "Document queued for processing"
}
```

**Error Responses**

| Status | Reason |
|--------|--------|
| `400` | Unsupported file type or file too large |
| `401` | Missing or invalid JWT token |
| `422` | Malformed request |
| `429` | Rate limit exceeded |

---

### 6.2 POST `/query`

Run a RAG query against an ingested collection.

**Request** — `application/json`

```json
{
  "question": "What was the revenue in Q3?",
  "collection_name": "user_123",
  "use_hyde": true,
  "top_k": 20,
  "top_n": 5
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `question` | string | Yes | — | User question in any language |
| `collection_name` | string | No | user ID | Qdrant collection to search |
| `use_hyde` | boolean | No | `true` | Enable HyDE query enhancement |
| `top_k` | integer | No | `20` | Candidates before reranking |
| `top_n` | integer | No | `5` | Final chunks after reranking |

**Response** `200 OK` — streaming (`text/event-stream`)

```
data: {"token": "The"}
data: {"token": " revenue"}
data: {"token": " in Q3"}
...
data: {"sources": [{"type": "table", "page": 4}, {"type": "text", "page": 5}]}
data: [DONE]
```

**Error Responses**

| Status | Reason |
|--------|--------|
| `401` | Missing or invalid JWT token |
| `404` | Collection not found |
| `429` | Rate limit exceeded |

---

### 6.3 GET `/health`

System health check — no authentication required.

**Response** `200 OK`

```json
{
  "status": "ok",
  "qdrant": "connected",
  "redis": "connected",
  "llm": "reachable"
}
```

---

### 6.4 POST `/auth/token`

Issue a JWT token (login).

**Request** — `application/x-www-form-urlencoded`

| Field | Type | Description |
|-------|------|-------------|
| `username` | string | Username |
| `password` | string | Password |

**Response** `200 OK`

```json
{
  "access_token": "eyJhbGci...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

---

## 7. RAG Pipeline — Data Flow

### 7.1 Ingestion (background task)

```
File Upload
    │
    ▼
File Validation (MIME type, size, extension)
    │
    ▼
Parser (by file type)
    ├── PDF  → PyMuPDF (text per page) + pdfplumber (tables) + PyMuPDF (images)
    ├── DOCX → python-docx (paragraphs + tables + embedded images)
    ├── TXT  → plain text read
    └── XLSX/CSV → pandas (full table per sheet)
    │
    ▼
Chunking Strategy
    ├── Text  → RecursiveCharacterTextSplitter (chunk_size=500, overlap=100)
    ├── Table → one atomic chunk per table (never split), stored as markdown
    └── Image → stored as base64 in metadata, not chunked
    │
    ▼
Embedding (sentence-transformers multilingual-MiniLM)
    ├── Text chunks  → 384-dim normalized vector
    └── Table chunks → 384-dim normalized vector
    │  (Images are NOT embedded — passed directly to Groq vision LLM at query time)
    │
    ▼
Qdrant Upsert
    └── Payload metadata: {type, page, source, chunk_index, image_base64 (if image)}
```

### 7.2 Query (real-time)

```
User Question (any language)
    │
    ▼
Language handled natively by multilingual embedding model
    │
    ▼
HyDE (if enabled)
    ├── LLM generates a hypothetical answer paragraph via Groq
    ├── Embed hypothetical answer → 384-dim vector
    └── Average with original query embedding → enhanced query vector
    │
    ▼
Hybrid Search (parallel)
    ├── Qdrant vector search  → top-K by cosine similarity
    └── BM25 keyword search   → top-K by TF-IDF score
    │
    ▼
Reciprocal Rank Fusion (RRF)
    └── Merge and re-rank both result lists → unified top-K candidates
    │
    ▼
Cross-Encoder Reranking
    └── Rescore all top-K candidates → keep top-N (default 5)
    │
    ▼
Prompt Construction
    ├── Text chunks → added as plain context
    ├── Table chunks → added as markdown tables
    └── Image chunks → added as base64 image_url blocks
    │
    ▼
Groq LLM (Llama 4 Scout) — streaming
    │
    ▼
Streamed response + source citations → Chainlit UI
```

---

## 8. Document Parsing Specification

### 8.1 PDF

| Content Type | Tool | Method |
|-------------|------|--------|
| Text | PyMuPDF | `page.get_text()` per page |
| Tables | pdfplumber | `page.extract_tables()` — geometric boundary detection |
| Images | PyMuPDF | `page.get_images()` + `doc.extract_image()` — stored as base64 |

Table output format (stored in vector store):

```
| Column A | Column B | Column C |
|----------|----------|----------|
| value 1  | value 2  | value 3  |
```

Metadata stored per table chunk:

```json
{
  "type": "table",
  "page": 4,
  "table_index": 1,
  "source": "report.pdf",
  "headers": ["Column A", "Column B", "Column C"]
}
```

### 8.2 DOCX

| Content Type | Tool | Method |
|-------------|------|--------|
| Text | python-docx | `doc.paragraphs` — preserves heading hierarchy |
| Tables | python-docx | `doc.tables` — headers separated from data rows |
| Images | python-docx | `doc.part.rels` — embedded image blobs extracted |

### 8.3 Excel / CSV

| Content Type | Tool | Method |
|-------------|------|--------|
| Tabular data | pandas | `read_excel()` / `read_csv()` — each sheet = one table chunk |

### 8.4 TXT

Plain UTF-8 read → chunked by `RecursiveCharacterTextSplitter`.

---

## 9. Vector Store Specification (Qdrant)

### Collection Schema

**Collection name:** `{user_id}` (one collection per user, isolated)

**Vector config:**
- Dimension: `384` (multilingual-MiniLM output)
- Distance: `Cosine`

**Payload fields per point:**

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Chunk text content (or markdown table) |
| `type` | string | `"text"`, `"table"`, or `"image"` |
| `page` | integer | Source page number |
| `source` | string | Original filename |
| `chunk_index` | integer | Position within document |
| `table_index` | integer | Table number on the page (tables only) |
| `headers` | list[string] | Column headers (tables only) |
| `image_base64` | string | Base64-encoded PNG (images only) |

---

## 10. Embedding Specification

**Model:** `paraphrase-multilingual-MiniLM-L12-v2`

| Property | Value |
|----------|-------|
| Output dimension | 384 |
| Languages supported | 50+ |
| GPU required | No |
| Max input tokens | 128 |
| Normalization | L2 normalized (cosine similarity ready) |

Images are **not embedded** — they are stored as base64 in Qdrant payload and injected directly into the Groq vision LLM prompt at query time.

---

## 11. HyDE Specification

| Property | Value |
|----------|-------|
| Trigger | `use_hyde=true` in query request |
| Generator | Groq LLM (same model as generation) |
| Prompt | "Write a short factual paragraph that directly answers: {question}" |
| Embedding | Same multilingual-MiniLM model |
| Combination | Average of query embedding + hypothetical doc embedding, re-normalized |

---

## 12. Hybrid Search Specification

| Property | Value |
|----------|-------|
| Vector search | Qdrant cosine similarity, top-K results |
| Keyword search | BM25 (rank-bm25) over stored text payloads, top-K results |
| Merging | Reciprocal Rank Fusion — `score = Σ 1 / (k + rank)` where `k=60` |
| Output | Single merged ranked list, top-K candidates for reranking |

---

## 13. Reranking Specification

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`

| Property | Value |
|----------|-------|
| Input | Query + each retrieved chunk (full text) |
| Output | Relevance score per chunk |
| GPU required | No |
| Input limit | top-K candidates (default 20) |
| Output | top-N chunks (default 5) |

---

## 14. Security Specification

### Authentication
- JWT tokens signed with `HS256`
- Token passed as `Authorization: Bearer <token>` header
- All `/ingest` and `/query` endpoints require valid token
- `/health` is public

### Rate Limiting
- Tool: `slowapi`
- Default: `20 requests/minute` per user (configurable via `RATE_LIMIT` env var)
- Returns `429 Too Many Requests` when exceeded

### File Validation
- MIME type verified against extension
- Max file size: 50MB (configurable)
- Malformed or password-protected PDFs rejected with `400`

---

## 15. Async Processing Specification

**Tool:** Celery + Redis

| Property | Value |
|----------|-------|
| Broker | Redis (`REDIS_URL`) |
| Task | `tasks.ingest_task.process_document` |
| Trigger | `POST /ingest` enqueues the task and returns `task_id` immediately |
| Retries | 3 retries with exponential backoff on failure |
| Result storage | Redis (task status queryable by `task_id`) |

---

## 16. Observability Specification

### Logging (Loguru)
- Format: JSON structured logs
- Levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`
- Output: stdout + rotating file (`logs/app.log`, max 10MB, 7 backups)
- Every request logs: method, path, user_id, duration_ms, status_code

### Tracing (LangSmith)
- Every RAG pipeline run traced end-to-end
- Captured: query, HyDE output, retrieved chunks, reranked chunks, final prompt, LLM response, latency

### Evaluation (RAGAS)
- Metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall
- Run on a held-out evaluation set after ingestion

---

## 17. Non-Functional Requirements

| Requirement | Target |
|-------------|--------|
| Query latency (p95) | < 5 seconds end-to-end |
| Ingestion throughput | 1 document per worker concurrently |
| Vector store | Persistent across restarts |
| Uptime | 99.9% (Docker health checks + restart policy) |
| Max file size | 50 MB |
| Supported languages | 50+ (native, no translation) |
| GPU | Not required |
| Python version | 3.12 |

---

## 18. Setup

### Install uv

```bash
pip install uv
```

### Install dependencies

```bash
uv sync           # production dependencies
uv sync --group dev  # include dev tools
```

### Configure environment

```bash
cp .env.example .env
# Fill in GROQ_API_KEY, LANGSMITH_API_KEY, JWT_SECRET_KEY
```

### Start all services

```bash
docker-compose up --build
```

Starts: FastAPI app, Chainlit UI, Qdrant, Redis, Celery worker.

### Local dev (no Docker)

```bash
uv run chainlit run app.py
```

### Run tests

```bash
uv run pytest tests/
```

### Lint and format

```bash
uv run ruff check .
uv run ruff format .
uv run mypy .
```
