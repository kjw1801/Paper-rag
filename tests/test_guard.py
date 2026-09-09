import pytest

from backend.guard import RequestGuard, RequestLimitExceeded


class Clock:
    def __init__(self, value: float = 1_800_000_000.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def test_limits_every_request_globally_per_minute() -> None:
    """방문자를 구분하지 않으므로 분당 한도는 인스턴스 전체에 적용된다."""
    clock = Clock()
    guard = RequestGuard(per_minute=2, daily_limit=10, max_concurrent=1, clock=clock)

    guard.admit()
    guard.admit()
    with pytest.raises(RequestLimitExceeded) as caught:
        guard.admit()

    assert caught.value.retry_after == 60
    clock.value += 60
    guard.admit()


def test_limits_total_requests_per_utc_day() -> None:
    clock = Clock()
    guard = RequestGuard(per_minute=10, daily_limit=2, max_concurrent=1, clock=clock)

    guard.admit()
    guard.admit()
    with pytest.raises(RequestLimitExceeded, match="오늘의"):
        guard.admit()

    clock.value += 24 * 60 * 60
    guard.admit()


def test_limits_concurrent_model_calls() -> None:
    guard = RequestGuard(per_minute=5, daily_limit=50, max_concurrent=1)

    with (
        guard.concurrency_slot(),
        pytest.raises(RequestLimitExceeded) as caught,
        guard.concurrency_slot(),
    ):
        pass

    assert caught.value.retry_after == 3


def test_rejects_non_positive_limits() -> None:
    with pytest.raises(ValueError):
        RequestGuard(per_minute=0)
