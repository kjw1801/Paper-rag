from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from google.api_core import exceptions as api_errors

from backend.api import (
    app,
    get_service,
    get_stats_store,
    get_turnstile_verifier,
    get_visit_guard,
)
from backend.guard import RequestGuard
from backend.models import AskResponse, Source
from backend.stats import StatsStore, today_in_seoul
from backend.turnstile import TurnstileVerifier
from tests.conftest import FakeStatsStore


@pytest.fixture(autouse=True)
def disable_turnstile():
    app.dependency_overrides[get_turnstile_verifier] = lambda: TurnstileVerifier(
        secret_key=None,
        required=False,
    )
    yield
    app.dependency_overrides.pop(get_turnstile_verifier, None)


class StubService:
    def ask(self, _question: str, _top_k: int) -> AskResponse:
        return AskResponse(
            answer="제안 방법의 F-measure는 0.15435입니다.",
            sources=[Source(page=5, snippet="표 1", relevance=0.91, cited=True)],
            grounded=True,
            cited_pages=[5],
        )


def test_visit_endpoint_counts_without_returning_numbers(
    fake_stats: FakeStatsStore,
) -> None:
    with TestClient(app) as client:
        response = client.post("/stats/visit")
        numbers = client.get("/stats").json()

    assert response.status_code == 204
    assert fake_stats.visits == 1
    assert numbers["today_visits"] == 1
    assert numbers["started_at"] == "2026-09-17"


def test_successful_question_increments_the_question_counter(
    fake_stats: FakeStatsStore,
) -> None:
    app.dependency_overrides[get_service] = lambda: StubService()

    try:
        with TestClient(app) as client:
            client.post("/ask", json={"question": "F-measure는 얼마인가요?"})
    finally:
        app.dependency_overrides.pop(get_service, None)

    assert fake_stats.questions == 1


def test_answer_survives_a_broken_counter() -> None:
    """집계가 죽어도 답변은 나가야 한다. 숫자보다 답변이 중요하다."""

    class BrokenStatsStore(FakeStatsStore):
        def record_question(self) -> None:
            raise RuntimeError("집계 저장소 장애")

    app.dependency_overrides[get_service] = lambda: StubService()
    app.dependency_overrides[get_stats_store] = lambda: BrokenStatsStore()

    try:
        with TestClient(app) as client:
            response = client.post("/ask", json={"question": "F-measure는 얼마인가요?"})
    finally:
        app.dependency_overrides.pop(get_service, None)
        app.dependency_overrides.pop(get_stats_store, None)

    assert response.status_code == 200
    assert response.json()["grounded"] is True


def test_stats_endpoint_reports_unavailable_instead_of_zeros(
    fake_stats: FakeStatsStore,
) -> None:
    """읽지 못할 때 0을 보내면 화면이 집계가 0이라고 거짓말한다."""
    fake_stats.readable = False

    with TestClient(app) as client:
        response = client.get("/stats")

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "STATS_UNAVAILABLE"


def test_visit_limit_is_separate_from_the_question_limit() -> None:
    """방문 집계가 질문 예산을 먹으면 정작 질문이 막힌다."""
    guard = RequestGuard(per_minute=1, daily_limit=1, max_concurrent=1)
    app.dependency_overrides[get_visit_guard] = lambda: guard

    try:
        with TestClient(app) as client:
            assert client.post("/stats/visit").status_code == 204
            blocked = client.post("/stats/visit")

            app.dependency_overrides[get_service] = lambda: StubService()
            asked = client.post("/ask", json={"question": "F-measure는 얼마인가요?"})
    finally:
        app.dependency_overrides.pop(get_visit_guard, None)
        app.dependency_overrides.pop(get_service, None)

    assert blocked.status_code == 429
    assert blocked.json()["detail"]["code"] == "VISIT_RATE_LIMITED"
    assert asked.status_code == 200


def test_counter_stays_quiet_when_firestore_is_unreachable() -> None:
    """자격 증명이 없는 환경에서도 예외가 밖으로 새면 안 된다."""

    def unreachable():
        raise OSError("연결 불가")

    store = StatsStore("paper", client_factory=unreachable)

    store.record_visit()
    store.record_question()

    assert store.snapshot() is None


def test_aggregation_date_rolls_over_at_seoul_midnight() -> None:
    """UTC 15시가 KST 자정이다. UTC 기준으로 세면 하루가 9시간 어긋난다."""
    before = datetime(2026, 9, 17, 14, 59, tzinfo=UTC)
    after = datetime(2026, 9, 17, 15, 0, tzinfo=UTC)

    assert today_in_seoul(before) == "2026-09-17"
    assert today_in_seoul(after) == "2026-09-18"


def test_a_store_without_an_injected_client_cannot_reach_firestore() -> None:
    """격리 장치가 살아 있는지 확인한다. 뚫리면 운영 카운터가 오염된다."""
    store = StatsStore("paper")

    with pytest.raises(AssertionError, match="실제 Firestore"):
        store.record_visit()


class FakeSnapshot:
    def __init__(self, data: dict | None) -> None:
        self._data = data

    def to_dict(self) -> dict | None:
        return self._data


class FakeDocument:
    """create/get/set만 흉내 내는 Firestore 문서."""

    def __init__(self, data: dict | None = None) -> None:
        self.data = data
        self.merged: list[dict] = []
        self.day: FakeDocument | None = None

    def create(self, payload: dict) -> None:
        if self.data is not None:
            raise api_errors.AlreadyExists("이미 있습니다")
        self.data = dict(payload)

    def get(self) -> FakeSnapshot:
        return FakeSnapshot(self.data)

    def set(self, payload: dict, merge: bool = False) -> None:
        assert merge, "카운터 문서는 merge로만 써야 기존 값이 날아가지 않는다"
        self.merged.append(payload)
        self.data = {**(self.data or {}), **payload}

    def collection(self, _name: str) -> "FakeCollection":
        if self.day is None:
            return FakeCollection()
        return FakeCollection({today_in_seoul(): self.day})


class FakeCollection:
    def __init__(self, documents: dict[str, FakeDocument] | None = None) -> None:
        self.documents = documents if documents is not None else {}

    def document(self, name: str) -> FakeDocument:
        return self.documents.setdefault(name, FakeDocument())


class FakeBatch:
    def __init__(self) -> None:
        self.committed = False

    def set(self, _document: FakeDocument, _payload: dict, merge: bool = False) -> None:
        assert merge

    def commit(self) -> None:
        self.committed = True


class FakeClient:
    def __init__(
        self, service_document: FakeDocument, day: FakeDocument | None = None
    ) -> None:
        if day is not None:
            service_document.day = day
        self._collection = FakeCollection({"paper": service_document})

    def collection(self, _name: str) -> FakeCollection:
        return self._collection

    def batch(self) -> FakeBatch:
        return FakeBatch()


def test_started_at_is_repaired_when_the_document_lacks_it() -> None:
    """집계 배치만 먼저 성공하면 started_at 없는 문서가 남는다. 그대로 두면 영영 빈다."""
    service_document = FakeDocument({"total_visits": 3})
    store = StatsStore("paper", client_factory=lambda: FakeClient(service_document))

    store.record_visit()

    assert service_document.data is not None
    assert service_document.data["started_at"] == today_in_seoul()
    assert service_document.merged == [{"started_at": today_in_seoul()}]


def test_started_at_is_not_rewritten_when_it_already_exists() -> None:
    """매번 덮으면 집계 시작일이 계속 오늘로 밀린다."""
    service_document = FakeDocument({"started_at": "2026-01-01", "total_visits": 3})
    store = StatsStore("paper", client_factory=lambda: FakeClient(service_document))

    store.record_visit()

    assert service_document.data is not None
    assert service_document.data["started_at"] == "2026-01-01"
    assert service_document.merged == []


def test_snapshot_repairs_a_missing_started_at() -> None:
    """숫자는 있는데 시작일만 빈 문서는 조회하면서 스스로 고친다."""
    service_document = FakeDocument({"total_visits": 4, "total_questions": 1})
    store = StatsStore("paper", client_factory=lambda: FakeClient(service_document))

    snapshot = store.snapshot()

    assert snapshot is not None
    assert snapshot.started_at == today_in_seoul()
    assert service_document.merged == [{"started_at": today_in_seoul()}]


def test_snapshot_leaves_an_empty_counter_without_a_started_at() -> None:
    """아무도 다녀가지 않았는데 시작일부터 찍으면 집계가 시작된 척이 된다."""
    service_document = FakeDocument({})
    store = StatsStore("paper", client_factory=lambda: FakeClient(service_document))

    snapshot = store.snapshot()

    assert snapshot is not None
    assert snapshot.started_at is None
    assert service_document.merged == []


def test_snapshot_hides_the_date_when_the_repair_write_fails() -> None:
    """저장되지 않은 날짜를 보여주면 다음 조회에서 값이 달라진다."""

    class UnwritableDocument(FakeDocument):
        def set(self, payload: dict, merge: bool = False) -> None:
            raise OSError("쓰기 불가")

    service_document = UnwritableDocument({"total_visits": 4})
    store = StatsStore("paper", client_factory=lambda: FakeClient(service_document))

    snapshot = store.snapshot()

    assert snapshot is not None
    assert snapshot.total_visits == 4
    assert snapshot.started_at is None


def test_snapshot_repairs_when_only_the_daily_numbers_survive() -> None:
    """부모 문서를 지워도 days/ 하위는 남는다. 그때도 시작일을 복구해야 한다."""
    service_document = FakeDocument({})
    day_document = FakeDocument({"visits": 2})
    store = StatsStore(
        "paper", client_factory=lambda: FakeClient(service_document, day_document)
    )

    snapshot = store.snapshot()

    assert snapshot is not None
    assert snapshot.today_visits == 2
    assert snapshot.total_visits == 0
    assert snapshot.started_at == today_in_seoul()
    assert service_document.merged == [{"started_at": today_in_seoul()}]
