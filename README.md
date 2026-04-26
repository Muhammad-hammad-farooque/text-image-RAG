# Text-Image RAG

A multimodal Retrieval-Augmented Generation (RAG) system that processes documents containing text and images, then answers questions about them through a Chainlit chat interface powered by a vision-capable LLM.

## How It Works

1. **Document Loading** — upload a `.pdf`, `.docx`, or `.txt` file via the chat UI
2. **Embedding** — text chunks and images are embedded into a shared 512-dimensional vector space using OpenAI's CLIP model
3. **Indexing** — all embeddings are stored in a FAISS vector store for fast similarity search
4. **Retrieval** — on a query, the top-5 most relevant text chunks and images are retrieved using cosine similarity
5. **Generation** — retrieved context (text + images) is sent to a Groq-hosted vision LLM which streams the answer back

## Project Structure

```
text_image_rag/
├── backend.py                  # All RAG logic (CLIP, FAISS, document processing, LLM)
├── app.py                      # Chainlit frontend (UI only, imports from backend)
├── text_image_rag.ipynb        # Original development notebook (reference)
├── requirements.txt            # All dependencies
├── .env                        # API keys (not committed)
├── .gitignore
└── README.md
```

## Tech Stack

| Layer | Component | Tool |
|-------|-----------|------|
| Backend | Document parsing (PDF) | PyMuPDF (fitz) |
| Backend | Document parsing (DOCX) | python-docx |
| Backend | Embedding model | CLIP `openai/clip-vit-base-patch32` via HuggingFace |
| Backend | Deep learning runtime | PyTorch |
| Backend | Vector store | FAISS (faiss-cpu) |
| Backend | Orchestration & chunking | LangChain |
| Backend | LLM integration | LangChain-Groq |
| Backend | LLM | `meta-llama/llama-4-scout-17b-16e-instruct` via Groq |
| Backend | Image processing | Pillow |
| Backend | Numerical operations | NumPy |
| Backend | Secret management | python-dotenv |
| Frontend | Chat UI framework | Chainlit |

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

### 3. Run the app

```bash
chainlit run app.py
```

This opens the chat UI at `http://localhost:8000`.

## Usage

1. Open `http://localhost:8000` in your browser
2. Upload a document (`.pdf`, `.docx`, or `.txt`) when prompted
3. Wait for processing — the app will confirm how many text chunks and images were extracted
4. Ask any question about your document — the answer streams back in real time with sources shown below

## Notebook (Reference Only)

`text_image_rag.ipynb` is the original development notebook used for building and testing the pipeline. It is not required to run the application — use `app.py` instead.

To test the notebook standalone, open it in Jupyter and run all cells. A file dialog will pop up to select your document.

## Requirements

- Python 3.11+
- Groq API key (free tier available)
- Internet connection on first run (downloads CLIP model ~600MB from HuggingFace)
