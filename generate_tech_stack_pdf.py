from fpdf import FPDF


class PDF(FPDF):
    def header(self):
        self.set_fill_color(30, 30, 30)
        self.rect(0, 0, 210, 20, 'F')
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(255, 255, 255)
        self.set_xy(0, 5)
        self.cell(0, 10, "Text-Image RAG - Tech Stack Reference", align="C")
        self.set_text_color(0, 0, 0)
        self.ln(18)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, title, color):
        self.set_fill_color(*color)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 9, f"  {title}", ln=True, fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def tech_entry(self, name, category, description, used_for):
        # Name
        self.set_fill_color(245, 245, 245)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(20, 80, 160)
        self.cell(0, 7, f"  {name}", ln=True, fill=True)

        # Category badge
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, f"  Category: {category}", ln=True)

        # Description
        self.set_font("Helvetica", "", 9)
        self.set_text_color(40, 40, 40)
        self.set_x(10)
        self.multi_cell(0, 5, f"  Description: {description}")

        # Used for
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(60, 120, 60)
        self.set_x(10)
        self.multi_cell(0, 5, f"  Used for: {used_for}")

        self.set_draw_color(220, 220, 220)
        self.line(10, self.get_y() + 1, 200, self.get_y() + 1)
        self.ln(4)


# ── Data ──────────────────────────────────────────────────────────────────────

backend_stack = [
    (
        "PyMuPDF (fitz)",
        "Document Parsing",
        "A high-performance Python library for reading and rendering PDF files. Built on top of the MuPDF C library.",
        "Extracting text page-by-page and pulling embedded images (with xref IDs) from PDF documents."
    ),
    (
        "python-docx",
        "Document Parsing",
        "Python library for creating and modifying Microsoft Word (.docx) files. Provides access to paragraphs, tables, and embedded media.",
        "Extracting paragraphs, table rows, and embedded images from DOCX documents."
    ),
    (
        "CLIP (openai/clip-vit-base-patch32)",
        "Embedding Model",
        "Contrastive Language-Image Pretraining model by OpenAI. Maps both text and images into the same 512-dimensional vector space, enabling cross-modal similarity search.",
        "Generating embeddings for text chunks and images so both can be stored and searched in a unified FAISS vector store."
    ),
    (
        "HuggingFace Transformers",
        "ML Framework",
        "Industry-standard library for loading and running pretrained deep learning models including CLIP, BERT, LLaMA, and thousands more.",
        "Loading CLIPModel and CLIPProcessor from HuggingFace Hub and running inference for embedding generation."
    ),
    (
        "PyTorch",
        "Deep Learning Framework",
        "Open-source machine learning framework developed by Meta. Provides tensor operations, autograd, and neural network modules.",
        "Running CLIP model inference inside torch.no_grad() context for efficient embedding computation without gradient tracking."
    ),
    (
        "FAISS (faiss-cpu)",
        "Vector Store",
        "Facebook AI Similarity Search - a library for efficient similarity search and clustering of dense vectors. Supports millions of vectors with millisecond retrieval.",
        "Storing all text and image embeddings and performing fast cosine similarity search to retrieve the top-k most relevant chunks for a given query."
    ),
    (
        "LangChain",
        "Orchestration Framework",
        "Framework for building LLM-powered applications. Provides abstractions for documents, text splitters, vector stores, chat models, and message types.",
        "RecursiveCharacterTextSplitter for chunking text, Document objects for structured storage, FAISS integration, and HumanMessage for building multimodal LLM messages."
    ),
    (
        "LangChain-Groq",
        "LLM Integration",
        "Official LangChain integration package for Groq's API. Enables using Groq-hosted models through LangChain's standard chat model interface including streaming.",
        "Connecting to Groq's API via init_chat_model() and streaming responses token by token using llm.astream()."
    ),
    (
        "Groq API (meta-llama/llama-4-scout-17b-16e-instruct)",
        "LLM Provider",
        "Groq provides ultra-fast LLM inference using custom LPU (Language Processing Unit) hardware. The Llama 4 Scout model is a 17B parameter vision-capable model from Meta, hosted on Groq.",
        "Generating final answers by processing retrieved text excerpts and base64-encoded images in a single multimodal prompt."
    ),
    (
        "NumPy",
        "Numerical Computing",
        "Fundamental Python library for numerical operations. Provides multi-dimensional arrays and mathematical functions.",
        "Converting list of CLIP embeddings into a NumPy array before passing to FAISS.from_embeddings()."
    ),
    (
        "Pillow (PIL)",
        "Image Processing",
        "Python Imaging Library - standard library for opening, converting, and saving images in various formats.",
        "Converting raw image bytes (from PDF/DOCX) to RGB PIL Images, and encoding them as PNG base64 strings for storage and LLM consumption."
    ),
    (
        "python-dotenv",
        "Configuration",
        "Loads environment variables from a .env file into os.environ at runtime.",
        "Securely loading GROQ_API_KEY from the .env file without hardcoding secrets in source code."
    ),
]

frontend_stack = [
    (
        "Chainlit",
        "Chat UI Framework",
        "Open-source Python framework for building production-ready conversational AI interfaces. Provides a ChatGPT-like UI with built-in support for file uploads, streaming, session management, and message formatting.",
        "Entire frontend - rendering the chat interface, handling file upload via AskFileMessage, displaying streaming LLM responses token-by-token, and managing per-user session state via cl.user_session."
    ),
    (
        "@cl.on_chat_start",
        "Chainlit Decorator",
        "Chainlit lifecycle hook that triggers once when a new user opens the chat. Used to initialize the session.",
        "Showing the welcome message, prompting file upload, calling load_document() from backend, and storing the vector store in the user session."
    ),
    (
        "@cl.on_message",
        "Chainlit Decorator",
        "Chainlit lifecycle hook that triggers every time the user sends a message in the chat.",
        "Reading the user query, calling retrieve() and build_message() from backend, streaming the LLM response, and appending source references."
    ),
    (
        "cl.user_session",
        "Session Management",
        "Chainlit's built-in per-user key-value session store. Each browser tab gets an isolated session, enabling multiple users simultaneously.",
        "Storing vector_store and image_data_stores per user so each user's document is isolated from other users."
    ),
    (
        "cl.AskFileMessage",
        "File Upload Component",
        "Chainlit built-in component that renders a file upload button inside the chat and waits for the user to select a file.",
        "Allowing users to upload their PDF, DOCX, or TXT document directly in the chat interface."
    ),
    (
        "llm.astream()",
        "Async Streaming",
        "LangChain's async streaming method that yields response chunks as they are generated by the LLM, instead of waiting for the full response.",
        "Streaming the LLM answer word-by-word to the chat using response_msg.stream_token(), giving a real-time typing effect."
    ),
]


# ── Build PDF ─────────────────────────────────────────────────────────────────

pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

# Intro
pdf.set_font("Helvetica", "", 10)
pdf.set_text_color(60, 60, 60)
pdf.multi_cell(
    0, 6,
    "This document describes every technology used in the Text-Image RAG project. "
    "The project is split into two layers: Backend (all AI/ML logic) and Frontend (Chainlit chat UI). "
    "Both layers communicate through clean function imports with no business logic in the frontend."
)
pdf.ln(5)

# Pipeline overview box
pdf.set_fill_color(235, 245, 255)
pdf.set_draw_color(100, 150, 220)
pdf.set_font("Helvetica", "B", 9)
pdf.set_text_color(30, 30, 30)
pdf.multi_cell(
    0, 6,
    "  Pipeline: Document (PDF/DOCX/TXT)  ->  CLIP Embeddings  ->  FAISS Vector Store  "
    "->  Similarity Retrieval  ->  Groq LLM (Llama 4 Scout)  ->  Chainlit Chat UI",
    fill=True, border=1
)
pdf.ln(6)

# Backend section
pdf.section_title(f"BACKEND  (backend.py)  -  {len(backend_stack)} technologies", (40, 90, 160))
for entry in backend_stack:
    pdf.tech_entry(*entry)

pdf.add_page()

# Frontend section
pdf.section_title(f"FRONTEND  (app.py)  -  {len(frontend_stack)} technologies", (50, 130, 80))
for entry in frontend_stack:
    pdf.tech_entry(*entry)

# Summary table
pdf.ln(4)
pdf.section_title("SUMMARY", (80, 80, 80))
pdf.set_font("Helvetica", "B", 9)
pdf.set_fill_color(220, 220, 220)
pdf.cell(60, 7, "Technology", border=1, fill=True)
pdf.cell(50, 7, "Role", border=1, fill=True)
pdf.cell(80, 7, "File", border=1, fill=True)
pdf.ln()

rows = [
    ("PyMuPDF", "PDF parsing", "backend.py"),
    ("python-docx", "DOCX parsing", "backend.py"),
    ("CLIP (ViT-B/32)", "Text + Image embedding", "backend.py"),
    ("HuggingFace Transformers", "Model loading", "backend.py"),
    ("PyTorch", "Tensor inference", "backend.py"),
    ("FAISS", "Vector similarity search", "backend.py"),
    ("LangChain", "Orchestration & chunking", "backend.py"),
    ("LangChain-Groq", "LLM integration", "backend.py"),
    ("Groq / Llama 4 Scout", "Answer generation", "backend.py"),
    ("NumPy", "Array operations", "backend.py"),
    ("Pillow", "Image processing", "backend.py"),
    ("python-dotenv", "Secret management", "backend.py"),
    ("Chainlit", "Chat UI framework", "app.py"),
    ("cl.user_session", "Per-user session store", "app.py"),
    ("cl.AskFileMessage", "File upload widget", "app.py"),
    ("llm.astream()", "Token streaming", "app.py"),
]

pdf.set_font("Helvetica", "", 8)
for i, (tech, role, file) in enumerate(rows):
    fill = i % 2 == 0
    pdf.set_fill_color(248, 248, 248) if fill else pdf.set_fill_color(255, 255, 255)
    pdf.cell(60, 6, tech, border=1, fill=True)
    pdf.cell(50, 6, role, border=1, fill=True)
    pdf.cell(80, 6, file, border=1, fill=True)
    pdf.ln()

output_path = "Tech_Stack_Reference.pdf"
pdf.output(output_path)
print(f"PDF created: {output_path}")
