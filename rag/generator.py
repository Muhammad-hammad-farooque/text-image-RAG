from typing import Iterator

from groq import Groq

from core.config import settings
from core.logger import logger

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=settings.groq_api_key)
    return _client


def _build_context(chunks: list[dict]) -> tuple[list[str], list[dict], list[dict]]:
    text_parts: list[str] = []
    image_blocks: list[dict] = []
    sources: list[dict] = []

    for chunk in chunks:
        payload = chunk["payload"]
        chunk_type = payload.get("type", "text")

        if chunk_type == "image":
            image_b64 = payload.get("image_base64", "")
            if image_b64:
                image_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    }
                )
        else:
            text_parts.append(payload.get("text", ""))

        sources.append(
            {
                "type": chunk_type,
                "page": payload.get("page", 0),
                "source": payload.get("source", ""),
            }
        )

    return text_parts, image_blocks, sources


def generate_stream(question: str, chunks: list[dict]) -> Iterator[str | dict]:
    """
    Stream answer tokens. Each yield is either a str token or a final
    dict {"sources": [...]} after the last token.
    """
    client = _get_client()
    text_parts, image_blocks, sources = _build_context(chunks)

    context_text = (
        "\n\n---\n\n".join(text_parts) if text_parts else "No relevant context found."
    )

    system_prompt = (
        "You are a precise assistant that answers only from the provided context. "
        "If the answer is not in the context, state that clearly. "
        "For tables, reference column names explicitly. "
        "Cite the source document and page when possible."
    )

    user_content: list[dict] = [
        {"type": "text", "text": f"Context:\n{context_text}\n\nQuestion: {question}"}
    ] + image_blocks

    stream = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        stream=True,
        max_tokens=1024,
        temperature=0.2,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content

    logger.info(f"Generation complete ({len(chunks)} context chunks)")
    yield {"sources": sources}
