from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import FakeListChatModel

from backend.service import NO_EVIDENCE_ANSWER, RAGService


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


def test_answer_contains_page_grounded_sources() -> None:
    document = Document(
        page_content="그룹 추천의 F-measure는 0.15435이다.",
        metadata={"page": 5},
    )
    service = RAGService(
        StubVectorStore([(document, 0.82)]),
        FakeListChatModel(
            responses=["그룹 추천의 F-measure는 0.15435입니다. (PDF 5페이지)"]
        ),
    )

    response = service.ask("그룹 추천 성능은?")

    assert response.grounded is True
    assert response.sources[0].page == 5
    assert response.sources[0].relevance == 0.82


def test_low_relevance_question_does_not_call_llm() -> None:
    document = Document(page_content="추천 시스템", metadata={"page": 1})
    never_called: Any = FakeListChatModel(responses=[])
    service = RAGService(
        StubVectorStore([(document, 0.12)]),
        never_called,
        min_relevance=0.35,
    )

    response = service.ask("저자의 오늘 점심 메뉴는?")

    assert response.grounded is False
    assert response.answer == NO_EVIDENCE_ANSWER
    assert response.sources == []
