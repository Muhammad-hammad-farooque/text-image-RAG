import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from core.config import settings
from core.logger import logger

_tokenizer: AutoTokenizer | None = None
_model: AutoModelForSequenceClassification | None = None


def _load() -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        logger.info(f"Loading reranker: {settings.reranker_model}")
        _tokenizer = AutoTokenizer.from_pretrained(settings.reranker_model)
        _model = AutoModelForSequenceClassification.from_pretrained(settings.reranker_model)
        _model.eval()
        logger.info("Reranker loaded")
    return _tokenizer, _model


def rerank(query: str, chunks: list[dict], top_n: int) -> list[dict]:
    """Score each chunk against the query and return the top_n by relevance."""
    if not chunks:
        return []

    tokenizer, model = _load()

    pairs = [[query, chunk["payload"]["text"]] for chunk in chunks]
    encoded = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(**encoded).logits

    scores = logits.squeeze(-1).tolist()
    if isinstance(scores, float):
        scores = [scores]

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = score

    ranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    logger.info(f"Reranked {len(chunks)} candidates → top {top_n}")
    return ranked[:top_n]
