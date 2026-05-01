from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM & Tracing
    groq_api_key: str
    langsmith_api_key: str | None = None  # optional — enables LangSmith tracing when set

    # Infrastructure
    qdrant_url: str = "http://localhost:6333"
    redis_url: str = "redis://localhost:6379"

    # Auth
    jwt_secret_key: str = "dev-secret-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # Models
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"

    # RAG
    chunk_size: int = 500
    chunk_overlap: int = 100
    retrieval_top_k: int = 20
    rerank_top_n: int = 5
    max_file_size_mb: int = 50
    rate_limit: str = "20/minute"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
