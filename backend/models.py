from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    top_k: int = Field(default=4, ge=1, le=8)


class Source(BaseModel):
    page: int
    snippet: str
    relevance: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    grounded: bool


class HealthResponse(BaseModel):
    status: str
    api_key_configured: bool
    index_ready: bool
