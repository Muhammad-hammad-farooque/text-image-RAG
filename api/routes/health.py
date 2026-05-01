import redis as redis_lib
from fastapi import APIRouter
from groq import Groq
from qdrant_client import QdrantClient

from core.config import settings

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    result: dict[str, str] = {"status": "ok"}

    try:
        QdrantClient(url=settings.qdrant_url, timeout=2).get_collections()
        result["qdrant"] = "connected"
    except Exception:
        result["qdrant"] = "unreachable"
        result["status"] = "degraded"

    try:
        r = redis_lib.from_url(settings.redis_url, socket_connect_timeout=2)
        r.ping()
        result["redis"] = "connected"
    except Exception:
        result["redis"] = "unreachable"
        result["status"] = "degraded"

    try:
        Groq(api_key=settings.groq_api_key).models.list()
        result["llm"] = "reachable"
    except Exception:
        result["llm"] = "unreachable"
        result["status"] = "degraded"

    return result
