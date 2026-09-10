import pytest
from fastapi.testclient import TestClient

from backend.api import app, get_request_guard, get_service, get_turnstile_verifier
from backend.guard import RequestGuard
from backend.models import AskResponse, Source
from backend.service import UpstreamRateLimited
from backend.turnstile import TurnstileVerifier


@pytest.fixture(autouse=True)
def disable_turnstile_for_api_tests():
    app.dependency_overrides[get_turnstile_verifier] = lambda: TurnstileVerifier(
        secret_key=None,
        required=False,
    )
    yield
    app.dependency_overrides.clear()


class StubService:
    def ask(self, _question: str, _top_k: int) -> AskResponse:
        return AskResponse(
            answer="제안 방법의 F-measure는 0.15435입니다.",
            sources=[
                Source(
                    page=5, snippet="표 1 추천 성능 평가", relevance=0.91, cited=True
                )
            ],
            grounded=True,
            cited_pages=[5],
        )


def test_ask_endpoint_returns_answer_and_pdf_page() -> None:
    app.dependency_overrides[get_service] = lambda: StubService()

    try:
        with TestClient(app) as client:
            response = client.post(
                "/ask",
                json={"question": "제안 방법의 성능은?", "top_k": 4},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["sources"][0]["page"] == 5
    assert body["cited_pages"] == [5]


def test_health_reports_document_count() -> None:
    with TestClient(app) as client:
        body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["document_count"] >= 0
    assert body["index_ready"] == (body["document_count"] > 0)
    # 키 설정 여부는 외부에 알릴 이유가 없다
    assert "api_key_configured" not in body


class RateLimitedService:
    def ask(self, _question: str, _top_k: int) -> AskResponse:
        raise UpstreamRateLimited("Gemini API 호출 한도를 초과했습니다.")


def test_upstream_rate_limit_becomes_429() -> None:
    app.dependency_overrides[get_service] = lambda: RateLimitedService()
    try:
        with TestClient(app) as client:
            response = client.post("/ask", json={"question": "질문입니다"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 429
    assert response.json()["detail"]["code"] == "AI_RATE_LIMITED"
    assert "한도" in response.json()["detail"]["message"]


def test_request_guard_becomes_429_with_retry_after() -> None:
    guard = RequestGuard(per_minute=1, daily_limit=10, max_concurrent=1)
    app.dependency_overrides[get_service] = lambda: StubService()
    app.dependency_overrides[get_request_guard] = lambda: guard

    try:
        with TestClient(app) as client:
            first = client.post("/ask", json={"question": "첫 번째 질문입니다"})
            second = client.post(
                "/ask",
                json={"question": "두 번째 질문입니다"},
                # 호출자가 IP 헤더를 바꿔도 인스턴스 전체 한도는 그대로 적용된다.
                headers={"x-forwarded-for": "203.0.113.99"},
            )
    finally:
        app.dependency_overrides.clear()

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.headers["retry-after"] == "60"
    assert second.json()["detail"]["code"] == "DEMO_RATE_LIMITED"


class BrokenService:
    def ask(self, _question: str, _top_k: int) -> AskResponse:
        raise KeyError("page")


def test_internal_defects_are_not_disguised_as_service_outage() -> None:
    """자체 결함은 503이 아니라 500으로 드러나야 로그에서 상류 장애와 구분된다."""
    app.dependency_overrides[get_service] = lambda: BrokenService()

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/ask", json={"question": "질문입니다"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500
