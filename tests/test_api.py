from fastapi.testclient import TestClient

from backend.api import app, get_service
from backend.models import AskResponse, Source


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
