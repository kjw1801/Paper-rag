import os
import re
import shutil
from pathlib import Path
from typing import Any

import chromadb
from chromadb.errors import ChromaError
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from backend.ingest import load_pdf_pages, split_pages
from backend.models import AskResponse, GroundedAnswer, Source

ROOT_DIR = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT_DIR / "data" / "paper.pdf"
INDEX_PATH = ROOT_DIR / "data" / "chroma"
COLLECTION_NAME = "paper"

NO_EVIDENCE_ANSWER = "제공된 논문에서 이 질문에 답할 근거를 찾지 못했습니다."
MISSING_INDEX_MESSAGE = (
    "Chroma 인덱스가 없거나 비어 있습니다. "
    "먼저 `uv run python scripts/build_index.py`를 실행해 주세요."
)

OUTPUT_PARSER = PydanticOutputParser(pydantic_object=GroundedAnswer)

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 논문 질의응답 도우미입니다.
아래에 제공된 논문 문맥만 근거로 한국어로 답하세요.
문맥에 없는 사실을 추측하거나 외부 지식을 사용하지 마세요.
문맥이 질문과 무관하거나 근거가 부족하면 has_evidence를 false로 두고,
answer에는 논문에서 근거를 찾지 못했다고만 적으세요.
근거가 있으면 핵심을 먼저 말하고, 사용한 문맥의 페이지 번호를 cited_pages에 넣으세요.

{format_instructions}""",
        ),
        ("human", "질문: {question}\n\n논문 문맥:\n{context}"),
    ]
).partial(format_instructions=OUTPUT_PARSER.get_format_instructions())


def api_key_configured() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))


def create_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"),
        output_dimensionality=768,
    )


def index_document_count(index_path: Path = INDEX_PATH) -> int:
    """저장된 인덱스의 문서 수. 인덱스가 없으면 0."""
    if not index_path.exists():
        return 0
    try:
        client = chromadb.PersistentClient(path=str(index_path))
        return client.get_collection(COLLECTION_NAME).count()
    except (ChromaError, ValueError):
        return 0


def build_index(*, rebuild: bool = False) -> int:
    """PDF를 청크로 나눠 임베딩하고 Chroma에 저장한다. 저장된 청크 수를 돌려준다."""
    if not api_key_configured():
        raise RuntimeError("GOOGLE_API_KEY 또는 GEMINI_API_KEY가 필요합니다.")

    if INDEX_PATH.exists():
        if not rebuild and index_document_count() > 0:
            return index_document_count()
        shutil.rmtree(INDEX_PATH)

    chunks = split_pages(load_pdf_pages(PDF_PATH))
    if not chunks:
        raise RuntimeError(f"PDF에서 텍스트를 추출하지 못했습니다: {PDF_PATH}")

    Chroma.from_documents(
        documents=chunks,
        embedding=create_embeddings(),
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},
        persist_directory=str(INDEX_PATH),
    )

    count = index_document_count()
    if count != len(chunks):
        raise RuntimeError(
            f"인덱스 저장이 불완전합니다. 청크 {len(chunks)}개 중 {count}개만 저장됐습니다."
        )
    return count


class RAGService:
    def __init__(
        self,
        vector_store: Any,
        llm: Any,
        *,
        min_relevance: float = 0.35,
    ) -> None:
        self.vector_store = vector_store
        self.llm = llm
        self.min_relevance = min_relevance

    @classmethod
    def from_environment(cls) -> "RAGService":
        if not api_key_configured():
            raise RuntimeError("GOOGLE_API_KEY 또는 GEMINI_API_KEY가 필요합니다.")

        if index_document_count() == 0:
            raise RuntimeError(MISSING_INDEX_MESSAGE)

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=create_embeddings(),
            persist_directory=str(INDEX_PATH),
        )

        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_CHAT_MODEL", "gemini-3.6-flash"),
            response_mime_type="application/json",
        )

        return cls(
            vector_store,
            llm,
            min_relevance=float(os.getenv("RAG_MIN_RELEVANCE", "0.60")),
        )

    def retrieve(self, question: str, top_k: int) -> list[tuple[Document, float]]:
        """임계값을 적용하지 않은 검색 결과. 점수 분포 확인에 사용한다."""
        matches = self.vector_store.similarity_search_with_relevance_scores(
            question,
            k=top_k,
        )
        return [(document, float(score)) for document, score in matches]

    def ask(self, question: str, top_k: int = 4) -> AskResponse:
        matches = [
            (document, score)
            for document, score in self.retrieve(question, top_k)
            if score >= self.min_relevance
        ]

        if not matches:
            return _no_evidence_response()

        context = "\n\n".join(
            f"[PDF {document.metadata['page']}페이지]\n{document.page_content}"
            for document, _ in matches
        )
        result = self._generate(question, context)

        retrieved_pages = {int(document.metadata["page"]) for document, _ in matches}
        cited_pages = sorted(
            {page for page in result.cited_pages if page in retrieved_pages}
        )

        if not result.has_evidence:
            return _no_evidence_response()

        sources = [
            Source(
                page=int(document.metadata["page"]),
                snippet=best_sentence(question, document.page_content),
                relevance=round(score, 4),
                cited=int(document.metadata["page"]) in cited_pages,
            )
            for document, score in matches
        ]

        return AskResponse(
            answer=result.answer.strip(),
            sources=sources,
            grounded=True,
            cited_pages=cited_pages,
        )

    def _generate(self, question: str, context: str) -> GroundedAnswer:
        chain = PROMPT | self.llm | OUTPUT_PARSER
        try:
            return chain.invoke({"question": question, "context": context})
        except OutputParserException as error:
            # JSON 형식이 깨진 경우: 원문을 답변으로 쓰되 근거 없음 문구가 있으면 차단
            raw = str(error.llm_output or "").strip()
            return GroundedAnswer(
                answer=raw or NO_EVIDENCE_ANSWER,
                has_evidence=bool(raw) and "근거를 찾지 못했습니다" not in raw,
            )


def _no_evidence_response() -> AskResponse:
    return AskResponse(
        answer=NO_EVIDENCE_ANSWER,
        sources=[],
        grounded=False,
        cited_pages=[],
    )


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。])\s+")


def _bigrams(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", text)
    return {compact[i : i + 2] for i in range(len(compact) - 1)}


def best_sentence(question: str, text: str, max_length: int = 280) -> str:
    """청크 안에서 질문과 가장 겹치는 문장을 근거 문장으로 고른다."""
    compact = " ".join(text.split())
    sentences = [s for s in _SENTENCE_BOUNDARY.split(compact) if s]
    if not sentences:
        return ""

    question_grams = _bigrams(question)
    scored = [
        (len(question_grams & _bigrams(sentence)), index)
        for index, sentence in enumerate(sentences)
    ]
    _, best_index = max(scored)

    # 문장이 짧으면 다음 문장까지 이어 붙여 문맥을 보여준다
    snippet = sentences[best_index]
    next_index = best_index + 1
    while len(snippet) < 120 and next_index < len(sentences):
        snippet = f"{snippet} {sentences[next_index]}"
        next_index += 1

    if len(snippet) <= max_length:
        return snippet
    return f"{snippet[:max_length].rstrip()}..."
