# Text-Image RAG

A multimodal Retrieval-Augmented Generation (RAG) system that processes documents containing both text and images, then answers questions about them using a vision-capable LLM.

## How It Works

1. **Document Loading** — supports `.pdf`, `.docx`, and `.txt` files
2. **Embedding** — text and images are embedded into a shared vector space using OpenAI's CLIP model (`clip-vit-base-patch32`)
3. **Indexing** — all embeddings are stored in a FAISS vector store for fast similarity search
4. **Retrieval** — on a query, the top-k most relevant text chunks and images are retrieved
5. **Generation** — retrieved context (text + images) is passed to a Groq-hosted vision LLM which generates the final answer

## Tech Stack

| Component | Tool |
|-----------|------|
| Document parsing | PyMuPDF (PDF), python-docx (DOCX) |
| Embedding model | CLIP `openai/clip-vit-base-patch32` (via HuggingFace) |
| Vector store | FAISS |
| LLM | `meta-llama/llama-4-scout-17b-16e-instruct` via Groq |
| Orchestration | LangChain |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com).

### 3. Add your document

Place your document in the project root and update `file_path` in the notebook:

```python
file_path = "your_document.pdf"   # or .docx or .txt
```

### 4. Run the notebook

Open `text_image_rag.ipynb` in Jupyter and run all cells top to bottom.

## Usage

```python
answer = multimodal_pdf_rag_pipeline("What is the main topic of this document?")
print(answer)
```

## Requirements

- Python 3.11+
- Groq API key (free tier available)
- Internet connection (first run downloads the CLIP model ~600MB)
