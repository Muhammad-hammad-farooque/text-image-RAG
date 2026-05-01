import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from core.config import settings
from core.logger import logger

_tokenizer: AutoTokenizer | None = None
_model: AutoModel | None = None


def _load() -> tuple[AutoTokenizer, AutoModel]:
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        _tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)
        _model = AutoModel.from_pretrained(settings.embedding_model)
        _model.eval()
        logger.info("Embedding model loaded")
    return _tokenizer, _model


def _mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts. Returns shape (N, 384), L2-normalized."""
    tokenizer, model = _load()

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**encoded)

    embeddings = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.numpy()


def embed_text(text: str) -> np.ndarray:
    """Embed a single text. Returns shape (384,), L2-normalized."""
    return embed_texts([text])[0]
