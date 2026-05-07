import asyncio
import base64
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

        # Build page → [b64, ...] map and store in session for image display
        page_images: dict[int, list[str]] = {}
        for c in chunks:
            if c["type"] == "image":
                b64 = c["metadata"].get("image_base64", "")
                if b64:
                    page_images.setdefault(c["page"], []).append(b64)
        cl.user_session.set("page_images", page_images)

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


_DONE = object()


def _next_item(gen):
    try:
        return next(gen)
    except StopIteration:
        return _DONE


async def _astream_generate(question: str, chunks: list):
    loop = asyncio.get_running_loop()
    gen = generate_stream(question, chunks)
    while True:
        item = await loop.run_in_executor(None, _next_item, gen)
        if item is _DONE:
            break
        yield item


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

        # Collect pages from retrieved chunks and show matching images
        page_images: dict[int, list[str]] = cl.user_session.get("page_images") or {}
        if page_images:
            retrieved_pages = {
                chunk.get("payload", {}).get("page", -1)
                for chunk in ranked_chunks
            }
            image_elements = []
            seen: set[str] = set()
            for page in sorted(retrieved_pages):
                for b64 in page_images.get(page, []):
                    if b64 not in seen:
                        seen.add(b64)
                        image_elements.append(
                            cl.Image(
                                name=f"page{page}_{len(image_elements)}",
                                display="inline",
                                content=base64.b64decode(b64),
                            )
                        )

            if image_elements:
                await cl.Message(
                    content="**Relevant images from document:**",
                    elements=image_elements,
                ).send()

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
