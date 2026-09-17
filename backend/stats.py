"""공개 데모의 방문·질문 횟수를 Firestore에 숫자로만 쌓는다.

저장하는 것은 누적 숫자와 KST 날짜별 숫자뿐이다. 질문 내용, IP, 개별 접속
시각은 남기지 않는다.

집계는 전부 fail-open이다. Firestore가 죽더라도 답변은 그대로 나가야 하므로
모든 오류를 흡수하고 로그만 남긴다. 숫자 하나를 잃는 쪽이 답변을 잃는 쪽보다 낫다.

    services/{service}                 total_visits, total_questions, started_at
    services/{service}/days/{날짜}      visits, questions
"""

import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from google.api_core import exceptions as api_errors
from google.auth import exceptions as auth_errors
from google.cloud import firestore

logger = logging.getLogger(__name__)

# 자격 증명 부재, API 오류, 네트워크 단절까지 집계 실패로 본다.
# 이 밖의 예외는 자체 결함이므로 흡수하지 않고 드러낸다.
STORAGE_ERRORS = (
    api_errors.GoogleAPIError,
    auth_errors.GoogleAuthError,
    OSError,
)

KST = timezone(timedelta(hours=9))

SERVICES_COLLECTION = "services"
DAYS_COLLECTION = "days"


def today_in_seoul(now: datetime | None = None) -> str:
    """집계 기준 날짜. 브라우저가 보낸 날짜는 조작할 수 있으므로 서버에서 만든다."""
    moment = now if now is not None else datetime.now(KST)
    return moment.astimezone(KST).strftime("%Y-%m-%d")


@dataclass(frozen=True)
class StatsSnapshot:
    today_visits: int
    today_questions: int
    total_visits: int
    total_questions: int
    started_at: str | None


class StatsStore:
    """Firestore 카운터. 실패해도 예외를 밖으로 내보내지 않는다."""

    def __init__(
        self,
        service_id: str,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.service_id = service_id
        self._client_factory = client_factory
        self._lock = threading.Lock()
        self._client: Any | None = None
        self._unavailable = False
        self._started_at_ensured = False

    @classmethod
    def from_environment(cls) -> "StatsStore":
        return cls(os.getenv("RAG_STATS_SERVICE_ID", "paper"))

    def _connect(self) -> Any | None:
        """Firestore 클라이언트를 처음 쓸 때 만든다. 만들지 못하면 영구히 쉰다."""
        if self._unavailable:
            return None
        with self._lock:
            if self._client is None:
                try:
                    # 기본값으로 firestore.Client를 묶어 두면 import 시점에 고정돼
                    # 테스트가 이 이름을 바꿔치기해도 먹히지 않는다. 여기서 찾는다.
                    factory = self._client_factory or firestore.Client
                    self._client = factory()
                except STORAGE_ERRORS:
                    # 자격 증명이 없는 로컬 개발에서도 서비스는 그대로 떠야 한다.
                    logger.warning("통계 저장소를 사용할 수 없어 집계를 건너뜁니다.")
                    self._unavailable = True
                    return None
        return self._client

    def _service_document(self, client: Any) -> Any:
        return client.collection(SERVICES_COLLECTION).document(self.service_id)

    def _day_document(self, service_document: Any) -> Any:
        return service_document.collection(DAYS_COLLECTION).document(today_in_seoul())

    def _ensure_started_at(self, service_document: Any) -> None:
        """집계 시작일은 최초 한 번만 쓴다. merge로 매번 덮으면 날짜가 계속 밀린다."""
        if self._started_at_ensured:
            return
        try:
            service_document.create({"started_at": today_in_seoul()})
        except api_errors.AlreadyExists:
            # 집계 배치만 먼저 성공하면 문서는 있는데 started_at이 없을 수 있다.
            # 그대로 두면 영영 비게 되므로 실제로 있는지 확인하고 채운다.
            existing = service_document.get().to_dict() or {}
            if not existing.get("started_at"):
                service_document.set({"started_at": today_in_seoul()}, merge=True)
        except STORAGE_ERRORS:
            # 플래그를 세우지 않아 다음 요청에서 다시 시도한다.
            logger.warning("집계 시작일을 기록하지 못했습니다.", exc_info=True)
            return
        self._started_at_ensured = True

    def _increment(self, field: str) -> None:
        client = self._connect()
        if client is None:
            return
        try:
            service_document = self._service_document(client)
            self._ensure_started_at(service_document)
            # 누적과 일일을 한 배치로 올려 둘이 어긋나지 않게 한다.
            batch = client.batch()
            batch.set(
                service_document,
                {f"total_{field}": firestore.Increment(1)},
                merge=True,
            )
            batch.set(
                self._day_document(service_document),
                {field: firestore.Increment(1)},
                merge=True,
            )
            batch.commit()
        except STORAGE_ERRORS:
            logger.warning("통계 집계에 실패했습니다: %s", field, exc_info=True)

    def record_visit(self) -> None:
        self._increment("visits")

    def record_question(self) -> None:
        self._increment("questions")

    def snapshot(self) -> StatsSnapshot | None:
        """읽지 못하면 None. 화면에서 숫자 영역을 숨기는 신호로 쓴다."""
        client = self._connect()
        if client is None:
            return None
        try:
            service_document = self._service_document(client)
            service = service_document.get().to_dict() or {}
            day = self._day_document(service_document).get().to_dict() or {}
        except STORAGE_ERRORS:
            logger.warning("통계 조회에 실패했습니다.", exc_info=True)
            return None

        today_visits = int(day.get("visits", 0))
        today_questions = int(day.get("questions", 0))
        total_visits = int(service.get("total_visits", 0))
        total_questions = int(service.get("total_questions", 0))

        return StatsSnapshot(
            today_visits=today_visits,
            today_questions=today_questions,
            total_visits=total_visits,
            total_questions=total_questions,
            started_at=self._started_at(
                service_document,
                service.get("started_at"),
                # 부모 문서를 지워도 days/ 하위는 남는다. 누적만 보면 일일 숫자만
                # 남은 상태를 놓치므로 넷 다 본다.
                counted=any(
                    (today_visits, today_questions, total_visits, total_questions)
                ),
            ),
        )

    def _started_at(
        self, service_document: Any, stored: str | None, *, counted: bool
    ) -> str | None:
        """숫자는 있는데 시작일만 비어 있으면 이 자리에서 복구한다.

        집계 시작일은 프로세스 메모리의 플래그로 한 번만 쓰는데, 문서가 밖에서
        지워지면 그 프로세스는 다시 쓰지 않는다. 실제로 운영에서 한 번 그렇게 됐다.
        조회는 어차피 서비스 문서를 읽으므로 여기서는 추가 읽기가 들지 않는다.
        """
        if stored or not counted:
            return stored

        repaired = today_in_seoul()
        try:
            service_document.set({"started_at": repaired}, merge=True)
        except STORAGE_ERRORS:
            # 저장하지 못한 날짜를 보여주면 다음 조회에서 값이 달라진다. 숨긴다.
            logger.warning("집계 시작일을 복구하지 못했습니다.", exc_info=True)
            return None
        return repaired
