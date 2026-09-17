"""테스트가 실제 Firestore를 건드리지 않게 막는다.

개발 머신에는 gcloud 기본 자격 증명이 있어서, 막지 않으면 테스트 실행이
운영 카운터를 올린다. 실제로 한 번 그렇게 됐다.
"""

import pytest

from backend.api import app, get_stats_store
from backend.stats import StatsSnapshot


class FakeStatsStore:
    """숫자를 메모리에만 쌓는 카운터."""

    def __init__(self) -> None:
        self.visits = 0
        self.questions = 0
        self.readable = True

    def record_visit(self) -> None:
        self.visits += 1

    def record_question(self) -> None:
        self.questions += 1

    def snapshot(self) -> StatsSnapshot | None:
        if not self.readable:
            return None
        return StatsSnapshot(
            today_visits=self.visits,
            today_questions=self.questions,
            total_visits=self.visits,
            total_questions=self.questions,
            started_at="2026-09-17",
        )


@pytest.fixture
def fake_stats() -> FakeStatsStore:
    return FakeStatsStore()


def _refuse_real_firestore(*_args: object, **_kwargs: object):
    """StatsStore가 실제 Firestore를 잡으려 하면 시끄럽게 실패시킨다.

    AssertionError는 StatsStore의 STORAGE_ERRORS에 들어 있지 않으므로 흡수되지 않고
    테스트를 깨뜨린다. 조용히 건너뛰면 격리가 뚫린 줄 모르고 지나간다.
    """
    raise AssertionError(
        "테스트가 실제 Firestore에 접속하려 했습니다. 가짜 카운터를 주입하세요."
    )


@pytest.fixture(autouse=True)
def never_touch_real_firestore(
    fake_stats: FakeStatsStore, monkeypatch: pytest.MonkeyPatch
):
    # 의존성 주입을 우회해 StatsStore를 직접 만드는 테스트까지 막는다.
    monkeypatch.setattr("backend.stats.firestore.Client", _refuse_real_firestore)
    app.dependency_overrides[get_stats_store] = lambda: fake_stats
    yield
    app.dependency_overrides.pop(get_stats_store, None)
