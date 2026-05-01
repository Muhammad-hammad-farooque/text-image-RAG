import numpy as np
from rank_bm25 import BM25Okapi

from core.config import settings
from core.logger import logger
from rag.embedder import embed_text
from rag.hyde import enhance_query_vector
from rag.vector_store import get_all_texts, search as vector_search


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _rrf_merge(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """Reciprocal Rank Fusion: score = Σ 1 / (k + rank)."""
    scores: dict[str, float] = {}
    index: dict[str, dict] = {}

    for rank, result in enumerate(vector_results):
        text = result["payload"]["text"]
        scores[text] = scores.get(text, 0.0) + 1.0 / (k + rank + 1)
        index[text] = result

    for rank, result in enumerate(bm25_results):
        text = result["payload"]["text"]
        scores[text] = scores.get(text, 0.0) + 1.0 / (k + rank + 1)
        index[text] = result

    ranked_texts = sorted(scores, key=lambda t: scores[t], reverse=True)
    return [index[t] for t in ranked_texts]


def retrieve(
    question: str,
    collection_name: str,
    top_k: int | None = None,
    use_hyde: bool = True,
) -> list[dict]:
    top_k = top_k or settings.retrieval_top_k

    query_vector = embed_text(question)

    if use_hyde:
        try:
            query_vector = enhance_query_vector(query_vector, question)
        except Exception as e:
            logger.warning(f"HyDE failed, using plain query vector: {e}")

    vector_results = vector_search(query_vector, collection_name, top_k)
    logger.info(f"Vector search returned {len(vector_results)} results")

    # BM25 over all stored texts in the collection
    all_payloads = get_all_texts(collection_name)
    bm25_results: list[dict] = []
    if all_payloads:
        corpus = [p.get("text", "") for p in all_payloads]
        tokenized_corpus = [_tokenize(t) for t in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(_tokenize(question))
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        bm25_results = [
            {"score": float(bm25_scores[i]), "payload": all_payloads[i]}
            for i in top_indices
            if bm25_scores[i] > 0
        ]
        logger.info(f"BM25 returned {len(bm25_results)} results")

    merged = _rrf_merge(vector_results, bm25_results)
    logger.info(f"RRF merged → {len(merged)} candidates (capped at top {top_k})")
    return merged[:top_k]
