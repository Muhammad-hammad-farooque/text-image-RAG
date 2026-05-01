import numpy as np
from groq import Groq

from core.config import settings
from core.logger import logger

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=settings.groq_api_key)
    return _client


def _generate_hypothetical_doc(question: str) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Write a short factual paragraph that directly answers: {question}\n"
                    "Be concise and factual. Do not include preamble."
                ),
            }
        ],
        max_tokens=200,
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()


def enhance_query_vector(query_vector: np.ndarray, question: str) -> np.ndarray:
    """Average the query embedding with a hypothetical-answer embedding, then re-normalize."""
    from rag.embedder import embed_text  # late import to avoid circular dependency

    hyp_doc = _generate_hypothetical_doc(question)
    logger.info(f"HyDE hypothetical doc: {hyp_doc[:120]}...")

    hyp_vector = embed_text(hyp_doc)
    enhanced = (query_vector + hyp_vector) / 2.0

    norm = np.linalg.norm(enhanced)
    if norm > 0:
        enhanced = enhanced / norm
    return enhanced
