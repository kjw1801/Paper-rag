import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.guard import RequestGuard, RequestLimitExceeded
from backend.models import AskRequest, AskResponse, HealthResponse
from backend.service import (
    RAGService,
    UpstreamRateLimited,
    api_key_configured,
    build_index,
    index_document_count,
)

DEFAULT_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]


def allowed_origins() -> list[str]:
    """CORS_ALLOWED_ORIGINS=https://a.example,https://b.example 형태로 배포 주소를 추가한다."""
    configured = [
        origin.strip()
        for origin in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
        if origin.strip()
    ]
    return DEFAULT_ORIGINS + [o for o in configured if o not in DEFAULT_ORIGINS]


def build_index_on_startup() -> bool:
    return os.getenv("RAG_BUILD_INDEX_ON_STARTUP", "0").lower() in {"1", "true", "yes"}


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    # 배포 서버처럼 인덱스가 없는 환경에서는 시작 시 한 번 생성한다.
    if build_index_on_startup() and index_document_count() == 0:
        count = build_index()
        print(f"Chroma 인덱스를 생성했습니다 (청크 {count}개)")
    yield


app = FastAPI(
    title="Paper RAG API",
    description="논문 근거와 PDF 페이지를 함께 반환하는 RAG API",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@lru_cache(maxsize=1)
def get_service() -> RAGService:
    try:
        return RAGService.from_environment()
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


ServiceDependency = Annotated[RAGService, Depends(get_service)]


@lru_cache(maxsize=1)
def get_request_guard() -> RequestGuard:
    return RequestGuard.from_environment()


GuardDependency = Annotated[RequestGuard, Depends(get_request_guard)]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    count = index_document_count()
    return HealthResponse(
        status="ok",
        api_key_configured=api_key_configured(),
        index_ready=count > 0,
        document_count=count,
    )


@app.post("/ask", response_model=AskResponse)
def ask(
    payload: AskRequest,
    request: Request,
    service: ServiceDependency,
    guard: GuardDependency,
) -> AskResponse:
    forwarded_for = request.headers.get("x-forwarded-for", "")
    client_id = forwarded_for.split(",", maxsplit=1)[0].strip()
    if not client_id:
        client_id = request.client.host if request.client else "unknown"

    try:
        guard.admit(client_id)
        with guard.concurrency_slot():
            return service.ask(payload.question, payload.top_k)
    except RequestLimitExceeded as error:
        raise HTTPException(
            status_code=429,
            detail=str(error),
            headers={"Retry-After": str(error.retry_after)},
        ) from error
    except UpstreamRateLimited as error:
        raise HTTPException(status_code=429, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


def run() -> None:
    uvicorn.run(
        "backend.api:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RAG_RELOAD", "1") == "1",
    )
