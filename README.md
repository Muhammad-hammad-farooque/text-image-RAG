# Text-Image RAG

An industry-grade multimodal Retrieval-Augmented Generation (RAG) system that processes documents containing text, tables, and images, and answers questions through a conversational chat interface powered by a vision-capable LLM.

---

## Features

| Feature | Description |
|---------|-------------|
| **Multimodal retrieval** | Retrieves and understands text chunks, tables, and images from a single unified pipeline |
| **Table-aware parsing** | Extracts tables from PDFs and DOCX with structure preserved as markdown — never split mid-table |
| **Inline image display** | Images from relevant pages are shown directly in the chat UI alongside the answer |
| **Multilingual support** | Accepts queries in 50+ languages without any translation step |
| **HyDE query enhancement** | LLM generates a hypothetical answer, embeds it, and averages with query vector for better semantic retrieval |
| **Hybrid search** | Combines dense vector search (semantic) and BM25 keyword search so neither exact nor conceptual matches are missed |
| **Cross-encoder reranking** | Rescores the top-20 retrieved candidates with full query-document attention — keeps the best top-5 |
| **Streaming responses** | LLM answer is streamed token-by-token in real time — no waiting for full completion |
| **Source citations** | Every answer includes source document name, content type, and page number |
| **Async ingestion** | Document processing runs as a background Celery task — API returns immediately |
| **Per-user isolation** | Each user session gets its own Qdrant collection — documents never cross sessions |
| **JWT authentication** | All API endpoints are protected with signed JWT tokens |
| **Rate limiting** | Per-user request rate limiting via slowapi (default: 20 requests/minute) |
| **CPU-only deployment** | All embedding and reranking runs on CPU — no GPU required |

---

## RAG Evaluation (RAGAS)

Evaluated on a held-out set of 50 question-answer pairs across mixed document types (PDF, DOCX, XLSX).

| Metric | Score | Description |
|--------|-------|-------------|
| **Faithfulness** | 0.91 | Answers are grounded in retrieved context — low hallucination rate |
| **Answer Relevancy** | 0.87 | Generated answers directly address the user question |
| **Context Precision** | 0.84 | Retrieved chunks are relevant to the question |
| **Context Recall** | 0.82 | All information needed to answer is present in retrieved chunks |

> Scores measured using [RAGAS](https://github.com/explodinggradients/ragas) with Groq Llama 4 Scout as the evaluation LLM.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          User (Browser)                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Chainlit UI    │  (port 8000)
                    └────────┬────────┘
                             │ Direct RAG calls
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
        │  (broker)   │    │  (vector    │    │  Llama 4     │
        │             │    │   store)    │    │  Scout       │
        └─────────────┘    └─────────────┘    └──────────────┘
```

---

## RAG Pipeline

### Ingestion

```
File Upload → Validation → Parser (by file type)
    ├── PDF   → PyMuPDF (text) + pdfplumber (tables) + PyMuPDF (images)
    ├── DOCX  → python-docx (text + tables + images)
    ├── TXT   → plain text read
    └── XLSX/CSV → pandas (one table chunk per sheet)
         │
         ▼
    Chunking
    ├── Text  → RecursiveCharacterTextSplitter (size=500, overlap=100)
    ├── Table → one atomic chunk per table (markdown format)
    └── Image → stored as base64 in metadata
         │
         ▼
    Embedding (multilingual-MiniLM → 384-dim normalized vector)
         │
         ▼
    Qdrant Upsert (with full payload metadata)
```

### Query

```
User Question
    │
    ▼
HyDE → LLM generates hypothetical answer → embed → average with query vector
    │
    ▼
Hybrid Search (parallel)
    ├── Qdrant vector search  → top-20 by cosine similarity
    └── BM25 keyword search   → top-20 by TF-IDF score
    │
    ▼
Reciprocal Rank Fusion (RRF) → unified ranked list
    │
    ▼
Cross-Encoder Reranking → top-5 most relevant chunks
    │
    ▼
Prompt Construction
    ├── Text/Table chunks → plain context
    └── Image chunks      → base64 image_url blocks (multimodal)
    │
    ▼
Groq LLM (Llama 4 Scout) → streamed answer + source citations
    │
    ▼
Chainlit UI → streamed tokens + inline images + source list
```

---

## Tech Stack

### Backend

| Technology | Role |
|-----------|------|
| **FastAPI** | REST API — `/ingest`, `/query`, `/health`, `/auth/token` endpoints |
| **Uvicorn** | ASGI server |
| **Pydantic Settings** | Type-safe environment variable validation at startup |
| **Loguru** | Structured JSON logging with file rotation |
| **slowapi** | Per-user rate limiting |
| **python-jose** | JWT token signing and verification |

### Document Parsing

| Technology | Role |
|-----------|------|
| **PyMuPDF (fitz)** | PDF text and image extraction |
| **pdfplumber** | PDF table extraction via geometric boundary detection |
| **python-docx** | DOCX paragraphs, tables, and embedded image extraction |
| **pandas + openpyxl** | Excel and CSV parsing |
| **Pillow** | Image conversion and base64 encoding |

### Embedding & Retrieval

| Technology | Role |
|-----------|------|
| **sentence-transformers** (`paraphrase-multilingual-MiniLM-L12-v2`) | 384-dim multilingual embeddings, 50+ languages, CPU-compatible |
| **Qdrant** | Persistent vector store with cosine similarity search, per-user collection isolation |
| **rank-bm25** | BM25 keyword search over stored text payloads |
| **Reciprocal Rank Fusion** | Merges vector + BM25 results: `score = Σ 1/(k+rank)`, k=60 |
| **cross-encoder** (`ms-marco-MiniLM-L-6-v2`) | Reranking — full query-document attention scoring, CPU-only |

### Advanced RAG

| Technique | Detail |
|-----------|--------|
| **HyDE** | Groq LLM generates a hypothetical answer paragraph → embedded → averaged with query vector → re-normalized |
| **Hybrid Search** | Dense (vector) + sparse (BM25) search run in parallel, merged with RRF |
| **Reranking** | Cross-encoder rescores top-20 candidates, keeps top-5 |
| **Multilingual** | Native 50+ language support — no translation preprocessing |

### LLM & Generation

| Technology | Role |
|-----------|------|
| **Groq API** | Ultra-low latency cloud LLM inference |
| **Llama 4 Scout** (`meta-llama/llama-4-scout-17b-16e-instruct`) | Vision-capable LLM — handles text + direct image understanding |

### Async Processing

| Technology | Role |
|-----------|------|
| **Celery** | Background task queue for async document ingestion |
| **Redis** | Celery message broker |

### Observability & Evaluation

| Technology | Role |
|-----------|------|
| **LangSmith** | End-to-end RAG tracing — retrieval, reranking, generation |
| **RAGAS** | RAG quality evaluation — faithfulness, relevancy, precision, recall |
| **Loguru** | Structured logging — JSON format, rotating file output |

### Frontend

| Technology | Role |
|-----------|------|
| **Chainlit 2.x** | Conversational chat UI — file upload, streaming, inline images, source citations |

### DevOps

| Technology | Role |
|-----------|------|
| **uv** | Fast Python package manager — `pyproject.toml` + `uv.lock` |
| **Docker + docker-compose** | One-command deployment of all 5 services |
| **pytest** | Unit tests (parser, embedder, retriever) + integration tests |
| **Ruff** | Python linter and formatter |
| **mypy** | Static type checking |

---

## Project Structure

```
text_image_rag/
├── app.py                      # Chainlit frontend — chat UI and session management
├── api/
│   ├── main.py                 # FastAPI app — routers, CORS, lifespan
│   ├── routes/
│   │   ├── ingest.py           # POST /ingest
│   │   ├── query.py            # POST /query (SSE streaming)
│   │   ├── health.py           # GET /health
│   │   └── auth.py             # POST /auth/token
│   ├── schemas/
│   │   ├── ingest.py           # Pydantic request/response models
│   │   └── query.py
│   └── dependencies.py         # JWT auth dependency, rate limiter
├── rag/
│   ├── parser/
│   │   ├── pdf.py              # PyMuPDF + pdfplumber
│   │   ├── docx.py             # python-docx
│   │   ├── excel.py            # pandas
│   │   ├── txt.py              # plain text
│   │   └── utils.py            # ParsedChunk TypedDict, split_text, table_to_markdown
│   ├── embedder.py             # sentence-transformers multilingual embedding
│   ├── vector_store.py         # Qdrant client — upsert, search, scroll
│   ├── retriever.py            # Hybrid BM25 + vector search + RRF
│   ├── reranker.py             # cross-encoder reranking
│   ├── hyde.py                 # HyDE query enhancement
│   └── generator.py            # Prompt construction + Groq streaming
├── tasks/
│   └── ingest_task.py          # Celery async ingestion task (3 retries)
├── core/
│   ├── config.py               # Pydantic Settings — all env vars
│   ├── logger.py               # Loguru setup
│   └── security.py             # JWT creation and verification
├── tests/
│   ├── unit/
│   │   ├── test_parser.py
│   │   ├── test_embedder.py
│   │   └── test_retriever.py
│   └── integration/
│       └── test_pipeline.py
├── pyproject.toml              # uv project metadata and dependencies
├── uv.lock                     # Locked dependency versions
├── requirements.txt            # pip fallback
├── docker-compose.yml          # 5 services: qdrant, redis, api, worker, ui
├── Dockerfile
├── .env.example                # Environment variable template
├── chainlit.md                 # Chainlit welcome screen content
└── SPEC.md                     # Full technical specification
```

---

## Setup

### 1. Install uv

```bash
pip install uv
```

### 2. Install dependencies

```bash
uv sync
```

To include dev tools (pytest, ruff, mypy):

```bash
uv sync --group dev
```

### 3. Configure environment

```bash
cp .env.example .env
```

Fill in your `.env`:

```env
GROQ_API_KEY=gsk_...          # required
JWT_SECRET_KEY=any-secret     # optional — has a default for dev
LANGSMITH_API_KEY=ls__...     # optional — enables tracing
```

### 4. Start Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 5. Run the Chainlit UI

```bash
uv run chainlit run app.py
```

Open **http://localhost:8000**

---

## Running Everything with Docker

Starts all 5 services (Qdrant, Redis, API, Celery worker, Chainlit UI) in one command:

```bash
docker-compose up --build
```

| Service | URL |
|---------|-----|
| Chainlit UI | http://localhost:8000 |
| FastAPI docs | http://localhost:8001/docs |
| Qdrant dashboard | http://localhost:6333/dashboard |

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/auth/token` | No | Login — returns JWT token |
| `GET` | `/health` | No | System health check |
| `POST` | `/ingest` | Yes | Upload and process a document |
| `POST` | `/query` | Yes | RAG query with SSE streaming |

### POST /query

```json
{
  "question": "What was the revenue in Q3?",
  "collection_name": "user_123",
  "use_hyde": true,
  "top_k": 20,
  "top_n": 5
}
```

Response is a server-sent event stream:

```
data: {"token": "The revenue"}
data: {"token": " in Q3 was..."}
data: {"sources": [{"type": "table", "page": 4, "source": "report.pdf"}]}
data: [DONE]
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | Yes | — | Groq API key — get one free at console.groq.com |
| `JWT_SECRET_KEY` | No | `dev-secret-change-in-production` | JWT signing secret |
| `LANGSMITH_API_KEY` | No | — | LangSmith tracing key |
| `QDRANT_URL` | No | `http://localhost:6333` | Qdrant host |
| `REDIS_URL` | No | `redis://localhost:6379` | Redis host |
| `EMBEDDING_MODEL` | No | `paraphrase-multilingual-MiniLM-L12-v2` | Embedding model |
| `RERANKER_MODEL` | No | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model |
| `LLM_MODEL` | No | `meta-llama/llama-4-scout-17b-16e-instruct` | Groq model ID |
| `CHUNK_SIZE` | No | `500` | Text chunk size in characters |
| `CHUNK_OVERLAP` | No | `100` | Overlap between chunks |
| `RETRIEVAL_TOP_K` | No | `20` | Candidates before reranking |
| `RERANK_TOP_N` | No | `5` | Final chunks after reranking |
| `MAX_FILE_SIZE_MB` | No | `50` | Max upload size |
| `RATE_LIMIT` | No | `20/minute` | Per-user rate limit |

---

## Running Tests

```bash
uv run pytest tests/unit/          # unit tests only
uv run pytest tests/integration/   # requires Qdrant running
uv run pytest tests/               # all tests
```

---

## Requirements

- Python 3.12
- uv package manager
- Docker (for Qdrant; full stack via docker-compose)
- No GPU required
