import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.guard import RequestGuard, RequestLimitExceeded
from backend.models import AskRequest, AskResponse, HealthResponse
from backend.service import (
    RAGService,
    UpstreamRateLimited,
    UpstreamUnavailable,
    build_index,
    index_document_count,
)
from backend.turnstile import TurnstileRejected, TurnstileUnavailable, TurnstileVerifier

logger = logging.getLogger(__name__)

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
    # 배포 이미지에는 인덱스가 들어 있어 기본값은 생성하지 않는 것이다.
    # 켜 두더라도 생성 실패로 컨테이너가 죽지 않도록 흡수하고 /health가 준비 상태를 알린다.
    if build_index_on_startup() and index_document_count() == 0:
        try:
            count = build_index()
            logger.info("Chroma 인덱스를 생성했습니다 (청크 %d개)", count)
        except Exception:
            logger.exception("Chroma 인덱스 생성에 실패했습니다.")
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
        raise HTTPException(
            status_code=503,
            detail={
                "code": "SERVICE_NOT_READY",
                "message": "논문 검색 서비스를 준비 중입니다.",
            },
        ) from error


ServiceDependency = Annotated[RAGService, Depends(get_service)]


@lru_cache(maxsize=1)
def get_request_guard() -> RequestGuard:
    return RequestGuard.from_environment()


GuardDependency = Annotated[RequestGuard, Depends(get_request_guard)]


@lru_cache(maxsize=1)
def get_turnstile_verifier() -> TurnstileVerifier:
    return TurnstileVerifier.from_environment()


TurnstileDependency = Annotated[TurnstileVerifier, Depends(get_turnstile_verifier)]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    count = index_document_count()
    return HealthResponse(
        status="ok",
        index_ready=count > 0,
        document_count=count,
        turnstile_enabled=get_turnstile_verifier().enabled,
    )


@app.post("/ask", response_model=AskResponse)
def ask(
    payload: AskRequest,
    service: ServiceDependency,
    guard: GuardDependency,
    turnstile: TurnstileDependency,
) -> AskResponse:
    try:
        # 방문자 IP는 Cloud Run 프록시 뒤에서 신뢰할 수 없으므로 사용하지 않는다.
        turnstile.verify(payload.turnstile_token)
        guard.admit()
        with guard.concurrency_slot():
            return service.ask(payload.question, payload.top_k)
    except RequestLimitExceeded as error:
        raise HTTPException(
            status_code=429,
            detail={"code": "DEMO_RATE_LIMITED", "message": str(error)},
            headers={"Retry-After": str(error.retry_after)},
        ) from error
    except UpstreamRateLimited as error:
        raise HTTPException(
            status_code=429,
            detail={"code": "AI_RATE_LIMITED", "message": str(error)},
            headers={"Retry-After": "3600"},
        ) from error
    except UpstreamUnavailable as error:
        raise HTTPException(
            status_code=503,
            detail={"code": "AI_SERVICE_UNAVAILABLE", "message": str(error)},
        ) from error
    except TurnstileRejected as error:
        raise HTTPException(
            status_code=403,
            detail={"code": "TURNSTILE_REJECTED", "message": str(error)},
        ) from error
    except TurnstileUnavailable as error:
        raise HTTPException(
            status_code=503,
            detail={"code": "TURNSTILE_UNAVAILABLE", "message": str(error)},
        ) from error
    # 그 밖의 예외는 자체 결함이므로 500으로 드러내 로그에서 상류 장애와 구분한다.


def run() -> None:
    uvicorn.run(
        "backend.api:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RAG_RELOAD", "1") == "1",
    )
