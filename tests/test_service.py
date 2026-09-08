import json
from typing import Any

import pytest
from google.genai.errors import ClientError
from langchain_core.documents import Document
from langchain_core.language_models import FakeListChatModel
from langchain_google_genai._common import GoogleGenerativeAIError
from langchain_google_genai.chat_models import GoogleRateLimitError

from backend.service import (
    DEFAULT_MIN_RELEVANCE,
    NO_EVIDENCE_ANSWER,
    RAGService,
    UpstreamRateLimited,
    best_sentence,
)


class StubVectorStore:
    def __init__(self, matches: list[tuple[Document, float]]) -> None:
        self.matches = matches

    def similarity_search_with_relevance_scores(
        self,
        _question: str,
        *,
        k: int,
    ) -> list[tuple[Document, float]]:
        return self.matches[:k]


def fake_llm(**payload: Any) -> FakeListChatModel:
    return FakeListChatModel(responses=[json.dumps(payload, ensure_ascii=False)])


PAGE5 = Document(
    page_content="표 1은 추천 성능 평가 결과이다. 그룹 추천의 F-measure는 0.15435이다.",
    metadata={"page": 5},
)


def test_answer_contains_page_grounded_sources() -> None:
    service = RAGService(
        StubVectorStore([(PAGE5, 0.82)]),
        fake_llm(
            answer="F-measure는 0.15435입니다.", has_evidence=True, cited_pages=[5]
        ),
    )

    response = service.ask("그룹 추천 F-measure는?")

    assert response.grounded is True
    assert response.answer == "F-measure는 0.15435입니다."
    assert response.cited_pages == [5]
    assert response.sources[0].page == 5
    assert response.sources[0].relevance == 0.82
    assert response.sources[0].cited is True
    assert "0.15435" in response.sources[0].snippet


def test_cited_pages_are_limited_to_retrieved_pages() -> None:
    service = RAGService(
        StubVectorStore([(PAGE5, 0.8)]),
        fake_llm(answer="답", has_evidence=True, cited_pages=[5, 99]),
    )

    assert service.ask("질문").cited_pages == [5]


def test_model_without_evidence_is_not_grounded_even_with_matches() -> None:
    service = RAGService(
        StubVectorStore([(PAGE5, 0.8)]),
        fake_llm(answer="논문에 저자의 음식 취향은 없습니다.", has_evidence=False),
    )

    response = service.ask("저자가 좋아하는 음식은?")

    assert response.grounded is False
    assert response.answer == NO_EVIDENCE_ANSWER
    assert response.sources == []


def test_low_relevance_question_does_not_call_llm() -> None:
    never_called: Any = FakeListChatModel(responses=[])
    service = RAGService(
        StubVectorStore([(PAGE5, 0.12)]),
        never_called,
        min_relevance=DEFAULT_MIN_RELEVANCE,
    )

    response = service.ask("저자의 오늘 점심 메뉴는?")

    assert response.grounded is False
    assert response.answer == NO_EVIDENCE_ANSWER
    assert response.sources == []


def test_malformed_model_output_is_never_grounded() -> None:
    """JSON 형식을 어긴 답변은 검증할 수 없으므로 항상 근거 없음(fail-closed)."""
    for raw in ("F-measure는 0.15435입니다.", NO_EVIDENCE_ANSWER, "{broken json"):
        service = RAGService(
            StubVectorStore([(PAGE5, 0.8)]),
            FakeListChatModel(responses=[raw]),
        )

        response = service.ask("질문")

        assert response.grounded is False
        assert response.answer == NO_EVIDENCE_ANSWER
        assert response.sources == []


def test_has_evidence_without_valid_cited_pages_is_not_grounded() -> None:
    for cited in ([], [99]):
        service = RAGService(
            StubVectorStore([(PAGE5, 0.8)]),
            fake_llm(
                answer="근거 있는 척하는 답", has_evidence=True, cited_pages=cited
            ),
        )

        response = service.ask("질문")

        assert response.grounded is False
        assert response.answer == NO_EVIDENCE_ANSWER
        assert response.cited_pages == []


def test_best_sentence_picks_sentence_matching_question() -> None:
    text = (
        "협업 필터링은 사용자 간의 유사도를 측정한다. "
        "실험에는 사용자 100명과 거래 데이터 5000건을 사용하였다. "
        "그룹 추천 목록은 예측 값이 높은 순서로 생성한다."
    )

    snippet = best_sentence("실험 데이터의 사용자 수와 거래 데이터 수는?", text)

    assert snippet.startswith("실험에는 사용자 100명과 거래 데이터 5000건을")
    assert len(snippet) <= 280


class RateLimitedVectorStore:
    """임베딩 API가 429를 돌려줄 때 langchain-google-genai가 내는 형태를 흉내 낸다."""

    def similarity_search_with_relevance_scores(self, _q: str, *, k: int) -> list[Any]:
        try:
            raise ClientError(429, {"error": {"message": "RESOURCE_EXHAUSTED"}})
        except ClientError as cause:
            raise GoogleGenerativeAIError("Error embedding content") from cause


class RateLimitedChatModel(FakeListChatModel):
    def _call(self, *args: Any, **kwargs: Any) -> str:
        raise GoogleRateLimitError("429 RESOURCE_EXHAUSTED")


def test_rate_limit_during_retrieval_becomes_upstream_error() -> None:
    service = RAGService(RateLimitedVectorStore(), FakeListChatModel(responses=[]))

    with pytest.raises(UpstreamRateLimited):
        service.ask("질문")


def test_rate_limit_during_generation_becomes_upstream_error() -> None:
    service = RAGService(
        StubVectorStore([(PAGE5, 0.8)]), RateLimitedChatModel(responses=[])
    )

    with pytest.raises(UpstreamRateLimited):
        service.ask("질문")


def test_other_google_errors_are_not_masked_as_rate_limits() -> None:
    class BrokenVectorStore:
        def similarity_search_with_relevance_scores(
            self, _q: str, *, k: int
        ) -> list[Any]:
            raise GoogleGenerativeAIError("Error embedding content: invalid key")

    service = RAGService(BrokenVectorStore(), FakeListChatModel(responses=[]))

    with pytest.raises(GoogleGenerativeAIError):
        service.ask("질문")
