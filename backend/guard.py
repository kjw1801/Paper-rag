import math
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta


class RequestLimitExceeded(RuntimeError):
    def __init__(self, message: str, *, retry_after: int) -> None:
        super().__init__(message)
        self.retry_after = max(1, retry_after)


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError(f"{name}은 양의 정수여야 합니다: {raw}") from error
    if value < 1:
        raise RuntimeError(f"{name}은 1 이상이어야 합니다: {raw}")
    return value


class RequestGuard:
    """단일 Cloud Run 인스턴스용 저비용 요청 제한기.

    Cloud Run에서는 요청 소켓의 상대 주소가 구글 프런트엔드이고 X-Forwarded-For는
    호출자가 바꿔 보낼 수 있다. 신뢰할 수 없는 IP로 방문자를 구분하는 대신
    인스턴스 전체 한도만 적용하고, 사람 여부 확인은 Turnstile이 담당한다.
    """

    def __init__(
        self,
        *,
        per_minute: int = 15,
        daily_limit: int = 500,
        max_concurrent: int = 2,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if min(per_minute, daily_limit, max_concurrent) < 1:
            raise ValueError("요청 제한 값은 모두 1 이상이어야 합니다.")
        self.per_minute = per_minute
        self.daily_limit = daily_limit
        self._clock = clock
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(max_concurrent)
        self._global_requests: deque[float] = deque()
        self._day = self._utc_day(clock())
        self._daily_count = 0

    @classmethod
    def from_environment(cls) -> "RequestGuard":
        return cls(
            per_minute=_positive_int("RAG_RATE_LIMIT_PER_MINUTE", 15),
            daily_limit=_positive_int("RAG_DAILY_REQUEST_LIMIT", 500),
            max_concurrent=_positive_int("RAG_MAX_CONCURRENT_REQUESTS", 2),
        )

    @staticmethod
    def _utc_day(timestamp: float) -> date:
        return datetime.fromtimestamp(timestamp, UTC).date()

    def admit(self) -> None:
        now = self._clock()
        today = self._utc_day(now)
        with self._lock:
            if today != self._day:
                self._day = today
                self._daily_count = 0
                self._global_requests.clear()

            if self._daily_count >= self.daily_limit:
                tomorrow = datetime.combine(
                    today + timedelta(days=1), datetime.min.time(), UTC
                ).timestamp()
                raise RequestLimitExceeded(
                    "오늘의 공개 데모 질문 한도를 모두 사용했습니다. 내일 다시 시도해 주세요.",
                    retry_after=math.ceil(tomorrow - now),
                )

            window_start = now - 60
            while self._global_requests and self._global_requests[0] <= window_start:
                self._global_requests.popleft()
            if len(self._global_requests) >= self.per_minute:
                raise RequestLimitExceeded(
                    "공개 데모의 분당 질문 한도를 사용했습니다. 잠시 후 다시 시도해 주세요.",
                    retry_after=math.ceil(self._global_requests[0] + 60 - now),
                )
            self._global_requests.append(now)
            self._daily_count += 1

    @contextmanager
    def concurrency_slot(self) -> Iterator[None]:
        if not self._slots.acquire(blocking=False):
            raise RequestLimitExceeded(
                "현재 다른 질문을 처리하고 있습니다. 잠시 후 다시 시도해 주세요.",
                retry_after=3,
            )
        try:
            yield
        finally:
            self._slots.release()
