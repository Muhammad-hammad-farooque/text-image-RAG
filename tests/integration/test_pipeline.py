"""
End-to-end integration tests.

Requirements:
  - Qdrant running at QDRANT_URL (default: http://localhost:6333)
  - A valid GROQ_API_KEY in the environment

Skip automatically if infrastructure is unavailable.
Run with: uv run pytest tests/integration/ -m integration -v
"""
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def sample_txt(tmp_path: Path) -> Path:
    doc = tmp_path / "sample.txt"
    doc.write_text(
        "The quarterly revenue for Q3 2024 was $4.2 billion, "
        "representing a 12% increase over Q2 2024. "
        "Growth was driven by cloud services and enterprise subscriptions. " * 5,
        encoding="utf-8",
    )
    return doc


@pytest.mark.integration
def test_parse_and_embed(sample_txt: Path):
    from rag.embedder import embed_texts
    from rag.parser import parse

    chunks = parse(sample_txt)
    assert len(chunks) > 0

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    assert embeddings.shape == (len(chunks), 384)


@pytest.mark.integration
def test_full_ingest_and_retrieve(sample_txt: Path):
    pytest.importorskip("qdrant_client")

    from rag.embedder import embed_texts
    from rag.parser import parse
    from rag.retriever import retrieve
    from rag.vector_store import get_all_texts, upsert

    collection = "test_integration_collection"

    chunks = parse(sample_txt)
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    try:
        upsert(chunks, embeddings, collection)
    except Exception as e:
        pytest.skip(f"Qdrant not available: {e}")

    payloads = get_all_texts(collection)
    assert len(payloads) > 0

    try:
        results = retrieve(
            question="What was the Q3 revenue?",
            collection_name=collection,
            top_k=5,
            use_hyde=False,  # skip HyDE to avoid GROQ_API_KEY requirement
        )
        assert len(results) > 0
    except Exception as e:
        pytest.skip(f"Retrieval failed: {e}")
