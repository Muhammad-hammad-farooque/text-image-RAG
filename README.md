# Text-Image RAG

An industry-grade multimodal Retrieval-Augmented Generation (RAG) system that processes documents containing text, tables, and images, and answers questions through a conversational chat interface powered by a vision-capable LLM.

---

## How It Works

1. **Document Ingestion** — upload a `.pdf`, `.docx`, `.txt`, `.xlsx`, or `.csv` file; processing runs as a background task
2. **Parsing** — text, tables, and images are extracted with structure preserved per document type
3. **Embedding** — text and tables embedded using a multilingual sentence-transformers model; images passed directly to the vision LLM
4. **Indexing** — all embeddings stored in a persistent Qdrant vector store with metadata
5. **Retrieval** — query goes through HyDE + hybrid search (vector + BM25) + reranking to find the most relevant chunks
6. **Generation** — retrieved context (text, tables, images) sent to Groq-hosted Llama 4 Scout; answer streamed back in real time

---

## Tech Stack

### Backend

| Technology | Role |
|-----------|------|
| **FastAPI** | REST API framework — exposes `/ingest`, `/query`, and `/health` endpoints; decouples UI from RAG logic |
| **Uvicorn** | ASGI server — runs the FastAPI application |
| **Pydantic Settings** | Type-safe configuration management — validates all environment variables at startup, fails fast on missing keys |
| **Loguru** | Structured logging — JSON-formatted logs with log levels and file rotation for production observability |

### Document Parsing

| Technology | Role |
|-----------|------|
| **PyMuPDF (fitz)** | PDF text extraction — reads selectable text from PDF pages |
| **pdfplumber** | PDF table extraction — detects table boundaries geometrically and returns structured rows and columns |
| **python-docx** | DOCX parsing — extracts paragraphs, headings, and preserves table structure with headers |
| **pandas + openpyxl** | Excel and CSV parsing — reads tabular data and converts it to structured markdown for indexing |

### Embedding & Retrieval

| Technology | Role |
|-----------|------|
| **sentence-transformers** (`paraphrase-multilingual-MiniLM-L12-v2`) | Text and table embedding — fast CPU-compatible model with native support for 50+ languages; no translation step required |
| **Qdrant** | Persistent vector store — stores embeddings with metadata, supports filtered search, multi-collection isolation per user/document, and future ColPali multi-vector indexing |
| **rank_bm25** | BM25 keyword search — complements vector search with exact keyword matching for hybrid retrieval |
| **Reciprocal Rank Fusion (RRF)** | Hybrid search merging — combines vector similarity scores and BM25 scores into a single ranked list |
| **cross-encoder** (`ms-marco-MiniLM-L-6-v2`) | Reranking — rescores the top-20 retrieved chunks to keep the most relevant top-5; runs efficiently on CPU |

### Advanced RAG Techniques

| Technology | Role |
|-----------|------|
| **HyDE (Hypothetical Document Embeddings)** | Query enhancement — the LLM generates a hypothetical answer to the query, that answer is embedded and used for retrieval instead of the raw query; improves semantic match |
| **Multilingual Retrieval** | Language support — the multilingual embedding model handles queries in 50+ languages natively; no translation preprocessing needed |
| **Hybrid Search** | Retrieval quality — combines dense vector search (semantic) and sparse BM25 (keyword) so neither exact nor conceptual matches are missed |
| **Reranking** | Precision boost — a cross-encoder rescores retrieved candidates with full query-document attention, significantly improving top-k precision |

### LLM & Generation

| Technology | Role |
|-----------|------|
| **Groq API** | LLM inference — ultra-low latency cloud inference for both generation and HyDE query expansion |
| **Llama 4 Scout** (`meta-llama/llama-4-scout-17b-16e-instruct`) | Vision-capable LLM — handles text generation and direct image understanding; images are passed as base64 without local GPU processing |
| **LangChain** | RAG orchestration — manages the retrieval chain, prompt construction, and LLM streaming |
| **LangChain-Groq** | Groq integration — connects LangChain to the Groq inference API |

### Async Processing & Caching

| Technology | Role |
|-----------|------|
| **Celery** | Async task queue — runs document ingestion as a background job so the API returns immediately and the UI is never blocked |
| **Redis** | Message broker + cache — serves as the Celery task broker and caches embeddings for repeated documents and LLM responses |

### Observability & Evaluation

| Technology | Role |
|-----------|------|
| **LangSmith** | LLM tracing — records every retrieval and generation step end-to-end for debugging retrieval quality and latency |
| **RAGAS** | RAG evaluation — measures faithfulness, answer relevancy, and context precision against ground truth |
| **Prometheus + Grafana** | Metrics and dashboards — tracks query latency, token usage, ingestion throughput, and retrieval timing |

### Security

| Technology | Role |
|-----------|------|
| **JWT (python-jose)** | Authentication — issues and validates signed tokens for per-user access control |
| **FastAPI OAuth2** | Auth flow — handles token-based login and protected route dependencies |
| **slowapi** | Rate limiting — enforces per-user request limits to prevent abuse |
| **File validation** | Upload guard — checks MIME type, file size, and document integrity before processing |

### Frontend

| Technology | Role |
|-----------|------|
| **Chainlit** | Chat UI — provides the conversational interface with file upload, streaming responses, and source citation display |

### DevOps

| Technology | Role |
|-----------|------|
| **uv** | Package manager — manages dependencies, virtual environments, and lock file via `pyproject.toml` and `uv.lock`; faster than pip |
| **Docker + docker-compose** | Containerization — packages the app, Qdrant, Redis, and Celery worker into a single `docker-compose up` workflow |
| **GitHub Actions** | CI/CD pipeline — runs lint, type check, and tests on every push; builds and deploys on merge to main |
| **pytest** | Testing — unit tests for parsing and chunking logic, integration tests for the full retrieval pipeline |
| **Ruff** | Linting and formatting — fast Python linter and formatter enforcing code style and catching common errors |
| **mypy** | Type checking — static analysis to catch type errors before runtime |

---

## Project Structure

```
text_image_rag/
├── app.py                  # Chainlit frontend
├── backend.py              # RAG pipeline (parsing, embedding, retrieval, generation)
├── pyproject.toml          # Project metadata and all dependencies (uv)
├── uv.lock                 # Locked dependency versions (uv)
├── requirements.txt        # Mirror of pyproject.toml dependencies (pip fallback)
├── docker-compose.yml      # App + Qdrant + Redis + Celery
├── Dockerfile              # Application container
├── .env.example            # Environment variable template
├── .python-version         # Pinned Python version for uv
├── .gitignore
└── README.md
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
# Fill in GROQ_API_KEY and LANGSMITH_API_KEY
```

### 4. Start all services

```bash
docker-compose up --build
```

This starts the API, Chainlit UI, Qdrant vector store, Redis, and Celery worker together.

### 5. Open the chat UI

```
http://localhost:8000
```

---

## Running Without Docker (Local Dev)

```bash
uv run chainlit run app.py
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Groq API key for LLM inference — get one free at console.groq.com |
| `LANGSMITH_API_KEY` | LangSmith key for tracing — get one at smith.langchain.com |
| `QDRANT_URL` | Qdrant host URL (default: `http://localhost:6333`) |
| `REDIS_URL` | Redis connection URL (default: `redis://localhost:6379`) |
| `JWT_SECRET_KEY` | Secret key for signing JWT tokens |

---

## Requirements

- Python 3.12 (pinned via `.python-version`)
- uv package manager
- Docker + docker-compose
- No GPU required — all embedding and reranking runs on CPU
