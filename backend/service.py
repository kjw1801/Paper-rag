import fcntl
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.shared_system_client import SharedSystemClient
from chromadb.errors import ChromaError
from dotenv import load_dotenv
from google.genai.errors import ClientError
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_google_genai._common import GoogleGenerativeAIError
from langchain_google_genai.chat_models import GoogleRateLimitError

from backend.ingest import load_pdf_pages, split_pages
from backend.models import AskResponse, GroundedAnswer, Source

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT_DIR / "data" / "paper.pdf"
# 배포 환경에서는 영구 디스크 경로를 RAG_INDEX_DIR로 지정한다.
INDEX_PATH = Path(os.getenv("RAG_INDEX_DIR", str(ROOT_DIR / "data" / "chroma")))
COLLECTION_NAME = "paper"
# scripts/evaluate.py 실측: 논문 내 질문 최고 점수 0.683~0.774, 범위 밖 0.502~0.667
DEFAULT_MIN_RELEVANCE = 0.60

NO_EVIDENCE_ANSWER = "제공된 논문에서 이 질문에 답할 근거를 찾지 못했습니다."
MISSING_INDEX_MESSAGE = (
    "Chroma 인덱스가 없거나 비어 있습니다. "
    "먼저 `uv run python scripts/build_index.py`를 실행해 주세요."
)

OUTPUT_PARSER = PydanticOutputParser(pydantic_object=GroundedAnswer)


class UpstreamRateLimited(RuntimeError):
    """Gemini 호출 한도(무료 티어 분당·일당 요청 수)를 넘었을 때."""


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


def build_index(
    *,
    rebuild: bool = False,
    embeddings: Any | None = None,
    index_path: Path = INDEX_PATH,
) -> int:
    """PDF를 청크로 나눠 임베딩하고 Chroma에 저장한다. 저장된 청크 수를 돌려준다.

    새 인덱스는 임시 디렉터리에 먼저 만들고 검증이 끝난 뒤 교체하므로,
    임베딩 생성이 실패해도 기존 인덱스는 남는다.
    """
    if embeddings is None:
        if not api_key_configured():
            raise RuntimeError("GOOGLE_API_KEY 또는 GEMINI_API_KEY가 필요합니다.")
        embeddings = create_embeddings()

    existing = index_document_count(index_path)
    if existing > 0 and not rebuild:
        return existing

    # 여러 worker가 동시에 시작해도 한 프로세스만 생성하도록 파일 잠금을 건다
    index_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = index_path.with_name(f"{index_path.name}.lock")
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            # 잠금을 기다리는 동안 다른 worker가 이미 만들었을 수 있다
            existing = index_document_count(index_path)
            if existing > 0 and not rebuild:
                return existing
            return _build_index_locked(embeddings, index_path)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _build_index_locked(embeddings: Any, index_path: Path) -> int:
    chunks = split_pages(load_pdf_pages(PDF_PATH))
    if not chunks:
        raise RuntimeError(f"PDF에서 텍스트를 추출하지 못했습니다: {PDF_PATH}")

    # chromadb는 경로별로 클라이언트를 캐시하므로 매번 새 임시 경로를 쓴다
    building_path = index_path.with_name(
        f"{index_path.name}.building-{uuid.uuid4().hex[:8]}"
    )
    building_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            collection_metadata={"hnsw:space": "cosine"},
            persist_directory=str(building_path),
        )
        count = index_document_count(building_path)
        if count != len(chunks):
            raise RuntimeError(
                f"인덱스 저장이 불완전합니다. 청크 {len(chunks)}개 중 {count}개만 저장됐습니다."
            )
    except Exception:
        shutil.rmtree(building_path, ignore_errors=True)
        raise

    old_path = index_path.with_name(f"{index_path.name}.old")
    shutil.rmtree(old_path, ignore_errors=True)
    if index_path.exists():
        index_path.rename(old_path)
    building_path.rename(index_path)
    shutil.rmtree(old_path, ignore_errors=True)
    # 같은 프로세스에서 이전 경로를 열었던 클라이언트가 옮겨진 DB를 계속 보지 않도록 캐시를 비운다
    SharedSystemClient.clear_system_cache()
    return count


class RAGService:
    def __init__(
        self,
        vector_store: Any,
        llm: Any,
        *,
        min_relevance: float = DEFAULT_MIN_RELEVANCE,
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
            min_relevance=float(
                os.getenv("RAG_MIN_RELEVANCE", str(DEFAULT_MIN_RELEVANCE))
            ),
        )

    def retrieve(self, question: str, top_k: int) -> list[tuple[Document, float]]:
        """임계값을 적용하지 않은 검색 결과. 점수 분포 확인에 사용한다."""
        matches = self.vector_store.similarity_search_with_relevance_scores(
            question,
            k=top_k,
        )
        return [(document, float(score)) for document, score in matches]

    def ask(self, question: str, top_k: int = 4) -> AskResponse:
        try:
            return self._ask(question, top_k)
        except GoogleGenerativeAIError as error:
            # 질문 임베딩(검색)과 답변 생성 어느 단계의 한도 초과든 429로 올린다
            if _is_rate_limited(error):
                raise UpstreamRateLimited(
                    "Gemini API 호출 한도를 초과했습니다. 잠시 후 다시 시도해 주세요."
                ) from error
            raise

    def _ask(self, question: str, top_k: int) -> AskResponse:
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

        # 모델이 근거가 있다고 해도 검색된 페이지를 인용하지 않으면 근거 없음으로 본다
        if not result.has_evidence or not cited_pages:
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
        except OutputParserException:
            # 형식을 어긴 답변은 검증할 수 없으므로 근거 없음으로 처리한다 (fail-closed)
            return GroundedAnswer(answer=NO_EVIDENCE_ANSWER, has_evidence=False)


def _is_rate_limited(error: GoogleGenerativeAIError) -> bool:
    """chat 모델은 GoogleRateLimitError를, 임베딩은 ClientError(429)를 감싼 일반 오류를 낸다."""
    if isinstance(error, GoogleRateLimitError):
        return True
    cause = error.__cause__
    if isinstance(cause, ClientError) and cause.code == 429:
        return True
    return "RESOURCE_EXHAUSTED" in str(error) or "429" in str(error)


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
