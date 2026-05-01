from pathlib import Path

from celery import Celery

from core.config import settings
from core.logger import logger

celery_app = Celery(
    "ingest",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def process_document(self, file_path: str, collection_name: str) -> dict:
    from rag.embedder import embed_texts
    from rag.parser import parse
    from rag.vector_store import upsert

    try:
        logger.info(f"[task:{self.request.id}] Ingesting '{file_path}' → '{collection_name}'")

        chunks = parse(file_path)
        logger.info(f"[task:{self.request.id}] Parsed {len(chunks)} chunks")

        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)

        upsert(chunks, embeddings, collection_name)

        Path(file_path).unlink(missing_ok=True)

        logger.info(
            f"[task:{self.request.id}] Done — {len(chunks)} chunks in '{collection_name}'"
        )
        return {"status": "done", "chunks": len(chunks), "collection": collection_name}

    except Exception as exc:
        logger.error(f"[task:{self.request.id}] Failed: {exc}")
        Path(file_path).unlink(missing_ok=True)
        raise self.retry(exc=exc)
