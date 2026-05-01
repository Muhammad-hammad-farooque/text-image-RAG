from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str
    collection_name: str | None = None
    use_hyde: bool = True
    top_k: int = Field(default=20, ge=1, le=100)
    top_n: int = Field(default=5, ge=1, le=20)


class Source(BaseModel):
    type: str
    page: int
    source: str
