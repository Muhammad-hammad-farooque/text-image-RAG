import asyncio
from pathlib import Path

import chainlit as cl

from core.config import settings
from core.logger import logger
from rag.embedder import embed_texts
from rag.generator import generate_stream
from rag.parser import parse
from rag.reranker import rerank
from rag.retriever import retrieve
from rag.vector_store import upsert

ACCEPT_MIME_TYPES = [
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "text/csv",
]


def _ingest_file(file_path: str, collection: str) -> list:
    chunks = parse(file_path)
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    upsert(chunks, embeddings, collection)
    return chunks


@cl.on_chat_start
async def start():
    await cl.Message(
        content=(
            "Welcome to **Text-Image RAG**!\n\n"
            "Upload a document (`.pdf`, `.docx`, `.txt`, `.xlsx`, `.csv`) to get started."
        )
    ).send()

    files = await cl.AskFileMessage(
        content="Upload your document:",
        accept=ACCEPT_MIME_TYPES,
        max_size_mb=settings.max_file_size_mb,
    ).send()

    if not files:
        await cl.Message(content="No file received. Please refresh and try again.").send()
        return

    file = files[0]
    collection = cl.context.session.id or "default"

    status_msg = cl.Message(content=f"Processing **{file.name}**...")
    await status_msg.send()

    try:
        chunks = await asyncio.to_thread(_ingest_file, file.path, collection)
        cl.user_session.set("collection", collection)

        n_text = sum(1 for c in chunks if c["type"] == "text")
        n_table = sum(1 for c in chunks if c["type"] == "table")
        n_image = sum(1 for c in chunks if c["type"] == "image")

        await cl.Message(
            content=(
                f"**{file.name}** is ready!\n\n"
                f"- Text chunks: **{n_text}**\n"
                f"- Tables: **{n_table}**\n"
                f"- Images: **{n_image}**\n\n"
                "Ask me anything about your document."
            )
        ).send()

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        await cl.Message(content=f"Error processing document: `{str(e)}`").send()


async def _astream_generate(question: str, chunks: list):
    """Wrap the sync generate_stream generator as an async generator."""
    loop = asyncio.get_running_loop()
    gen = generate_stream(question, chunks)
    while True:
        try:
            item = await loop.run_in_executor(None, next, gen)
            yield item
        except StopIteration:
            break


@cl.on_message
async def main(message: cl.Message):
    collection = cl.user_session.get("collection")
    if not collection:
        await cl.Message(content="Please upload a document first.").send()
        return

    try:
        candidates = await asyncio.to_thread(
            retrieve,
            message.content,
            collection,
            settings.retrieval_top_k,
            True,  # use_hyde
        )

        if not candidates:
            await cl.Message(content="No relevant context found in the document.").send()
            return

        ranked_chunks = await asyncio.to_thread(
            rerank, message.content, candidates, settings.rerank_top_n
        )

        response_msg = cl.Message(content="")
        await response_msg.send()

        sources: list[dict] = []
        async for item in _astream_generate(message.content, ranked_chunks):
            if isinstance(item, str):
                await response_msg.stream_token(item)
            elif isinstance(item, dict) and "sources" in item:
                sources = item["sources"]

        if sources:
            source_lines = [
                f"- [{s['type'].capitalize()}] **{s['source']}** (page {s['page']})"
                for s in sources
            ]
            response_msg.content += "\n\n---\n**Sources:**\n" + "\n".join(source_lines)
            await response_msg.update()

    except Exception as e:
        logger.error(f"Query failed: {e}")
        await cl.Message(content=f"Error: `{str(e)}`").send()
