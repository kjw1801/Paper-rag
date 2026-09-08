from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    top_k: int = Field(default=4, ge=1, le=8)
    turnstile_token: str | None = Field(default=None, min_length=1, max_length=2048)


class Source(BaseModel):
    page: int
    snippet: str
    relevance: float
    cited: bool = False


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    grounded: bool
    cited_pages: list[int] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    api_key_configured: bool
    index_ready: bool
    document_count: int
    turnstile_enabled: bool


class GroundedAnswer(BaseModel):
    """Gemini가 반환하는 구조화된 답변."""

    answer: str = Field(
        description="한국어 답변. 근거가 없으면 그 사실을 한 문장으로 설명"
    )
    has_evidence: bool = Field(
        description="제공된 논문 문맥만으로 질문에 답할 수 있으면 true, 아니면 false"
    )
    cited_pages: list[int] = Field(
        default_factory=list,
        description="답변에 실제로 사용한 문맥의 PDF 페이지 번호 목록",
    )
