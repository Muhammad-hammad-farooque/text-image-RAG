import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from api.dependencies import get_current_user, limiter
from api.schemas.ingest import IngestResponse
from core.config import settings
from core.logger import logger

router = APIRouter()

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".xlsx", ".csv"}


@router.post("/ingest", response_model=IngestResponse, status_code=202)
@limiter.limit(settings.rate_limit)
async def ingest(
    request: Request,
    file: UploadFile = File(...),
    collection_name: str = Form(default=None),
    current_user: str = Depends(get_current_user),
) -> IngestResponse:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: '{ext}'")

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {size_mb:.1f} MB (max {settings.max_file_size_mb} MB)",
        )

    tmp_path = Path(tempfile.gettempdir()) / f"{uuid.uuid4()}{ext}"
    tmp_path.write_bytes(content)

    collection = collection_name or current_user

    try:
        from tasks.ingest_task import process_document

        task = process_document.delay(str(tmp_path), collection)
        task_id = task.id
        logger.info(f"Queued task {task_id} for '{file.filename}' → '{collection}'")
    except Exception as e:
        # Celery / Redis not available — run synchronously as fallback
        logger.warning(f"Celery unavailable ({e}), running ingestion synchronously")
        _ingest_sync(str(tmp_path), collection)
        task_id = str(uuid.uuid4())

    return IngestResponse(
        task_id=task_id,
        status="queued",
        filename=file.filename or "",
        message="Document queued for processing",
    )


def _ingest_sync(file_path: str, collection_name: str) -> None:
    from rag.embedder import embed_texts
    from rag.parser import parse
    from rag.vector_store import upsert

    chunks = parse(file_path)
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    upsert(chunks, embeddings, collection_name)
    Path(file_path).unlink(missing_ok=True)
    logger.info(f"Sync ingestion complete: {len(chunks)} chunks → '{collection_name}'")
