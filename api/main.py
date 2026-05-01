from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.dependencies import limiter
from api.routes import auth, health, ingest, query
from core.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Text-Image RAG API starting up")
    yield
    logger.info("Text-Image RAG API shutting down")


app = FastAPI(
    title="Text-Image RAG API",
    version="1.0.0",
    description="Industry-grade multimodal RAG system",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(auth.router, prefix="/auth", tags=["auth"])
