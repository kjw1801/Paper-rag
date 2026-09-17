import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.guard import RequestGuard, RequestLimitExceeded, positive_int
from backend.models import AskRequest, AskResponse, HealthResponse, StatsResponse
from backend.service import (
    RAGService,
    UpstreamRateLimited,
    UpstreamUnavailable,
    build_index,
    index_document_count,
)
from backend.stats import StatsStore
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
def get_visit_guard() -> RequestGuard:
    """방문 집계 전용 제한기.

    질문 한도를 함께 쓰면 방문이 예산을 먹어 정작 질문이 막힌다. 방문은 Turnstile로
    막을 수 없어 조작을 완전히 차단하지는 못하고, 이 한도는 완화책일 뿐이다.
    """
    return RequestGuard(
        per_minute=positive_int("RAG_VISIT_RATE_LIMIT_PER_MINUTE", 60),
        daily_limit=positive_int("RAG_VISIT_DAILY_LIMIT", 5000),
        max_concurrent=positive_int("RAG_VISIT_MAX_CONCURRENT_REQUESTS", 8),
    )


VisitGuardDependency = Annotated[RequestGuard, Depends(get_visit_guard)]


@lru_cache(maxsize=1)
def get_stats_store() -> StatsStore:
    return StatsStore.from_environment()


StatsDependency = Annotated[StatsStore, Depends(get_stats_store)]


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
    stats: StatsDependency,
) -> AskResponse:
    try:
        # 방문자 IP는 Cloud Run 프록시 뒤에서 신뢰할 수 없으므로 사용하지 않는다.
        turnstile.verify(payload.turnstile_token)
        guard.admit()
        with guard.concurrency_slot():
            answer = service.ask(payload.question, payload.top_k)
        # 집계는 답변이 나온 뒤의 부수 작업이다. 저장소 구현이 무엇으로 바뀌든
        # 여기서 샌 예외가 답변을 삼키면 안 되므로 폭넓게 막는다.
        try:
            stats.record_question()
        except Exception:
            logger.warning("질문 집계에 실패했습니다.", exc_info=True)
        return answer
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


@app.post("/stats/visit", status_code=204)
def record_visit(guard: VisitGuardDependency, stats: StatsDependency) -> None:
    """방문을 세기만 한다.

    숫자를 함께 돌려주면 증가는 됐는데 조회만 실패했을 때 프런트가 재시도하면서
    같은 방문이 두 번 세어진다. 숫자는 별도 GET으로 읽는다.
    """
    try:
        guard.admit()
        with guard.concurrency_slot():
            stats.record_visit()
    except RequestLimitExceeded as error:
        raise HTTPException(
            status_code=429,
            detail={"code": "VISIT_RATE_LIMITED", "message": str(error)},
            headers={"Retry-After": str(error.retry_after)},
        ) from error


@app.get("/stats", response_model=StatsResponse)
def read_stats(stats: StatsDependency) -> StatsResponse:
    snapshot = stats.snapshot()
    if snapshot is None:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "STATS_UNAVAILABLE",
                "message": "통계를 불러오지 못했습니다.",
            },
        )
    return StatsResponse(
        today_visits=snapshot.today_visits,
        today_questions=snapshot.today_questions,
        total_visits=snapshot.total_visits,
        total_questions=snapshot.total_questions,
        started_at=snapshot.started_at,
    )


def run() -> None:
    uvicorn.run(
        "backend.api:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RAG_RELOAD", "1") == "1",
    )
