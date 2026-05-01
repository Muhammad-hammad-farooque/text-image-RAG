import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.dependencies import get_current_user, limiter
from api.schemas.query import QueryRequest
from core.config import settings
from core.logger import logger

router = APIRouter()


@router.post("/query")
@limiter.limit(settings.rate_limit)
async def query(
    request: Request,
    body: QueryRequest,
    current_user: str = Depends(get_current_user),
) -> StreamingResponse:
    from rag.generator import generate_stream
    from rag.reranker import rerank
    from rag.retriever import retrieve

    collection = body.collection_name or current_user

    # Run blocking retrieval + reranking in a thread pool
    try:
        candidates = await asyncio.to_thread(
            retrieve,
            body.question,
            collection,
            body.top_k,
            body.use_hyde,
        )
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if not candidates:
        raise HTTPException(
            status_code=404,
            detail="No results found. The collection may be empty or does not exist.",
        )

    ranked_chunks = await asyncio.to_thread(rerank, body.question, candidates, body.top_n)

    async def event_stream():
        try:
            for item in generate_stream(body.question, ranked_chunks):
                if isinstance(item, str):
                    yield f"data: {json.dumps({'token': item})}\n\n"
                elif isinstance(item, dict) and "sources" in item:
                    yield f"data: {json.dumps({'sources': item['sources']})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
