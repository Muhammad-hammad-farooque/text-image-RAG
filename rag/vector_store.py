import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from core.config import settings
from core.logger import logger
from rag.parser.utils import ParsedChunk

VECTOR_DIM = 384

_client: QdrantClient | None = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.qdrant_url)
    return _client


def ensure_collection(collection_name: str) -> None:
    client = _get_client()
    existing = {c.name for c in client.get_collections().collections}
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        logger.info(f"Created Qdrant collection: {collection_name}")


def upsert(chunks: list[ParsedChunk], embeddings: np.ndarray, collection_name: str) -> None:
    client = _get_client()
    ensure_collection(collection_name)

    points: list[PointStruct] = []
    for chunk, vector in zip(chunks, embeddings):
        payload = {
            "text": chunk["text"],
            "type": chunk["type"],
            "page": chunk["page"],
            "source": chunk["source"],
            **chunk["metadata"],
        }
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload=payload,
            )
        )

    client.upsert(collection_name=collection_name, points=points)
    logger.info(f"Upserted {len(points)} points into '{collection_name}'")


def search(query_vector: np.ndarray, collection_name: str, top_k: int) -> list[dict]:
    client = _get_client()
    try:
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
            with_payload=True,
        )
    except Exception as e:
        logger.warning(f"Vector search failed for '{collection_name}': {e}")
        return []
    return [{"score": r.score, "payload": r.payload} for r in results]


def get_all_texts(collection_name: str) -> list[dict]:
    """Return all point payloads — used to build the BM25 index."""
    client = _get_client()
    try:
        records, _ = client.scroll(
            collection_name=collection_name,
            limit=10_000,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        logger.warning(f"Scroll failed for '{collection_name}': {e}")
        return []
    return [r.payload for r in records if r.payload]
