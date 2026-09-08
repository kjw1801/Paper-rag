import os
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from backend.ingest import load_pdf_pages, split_pages
from backend.models import AskResponse, Source

ROOT_DIR = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT_DIR / "data" / "paper.pdf"
INDEX_PATH = ROOT_DIR / "data" / "chroma"

NO_EVIDENCE_ANSWER = "제공된 논문에서 이 질문에 답할 근거를 찾지 못했습니다."

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 논문 질의응답 도우미입니다.
아래에 제공된 논문 문맥만 근거로 한국어로 답하세요.
문맥에 없는 사실을 추측하거나 외부 지식을 사용하지 마세요.
답을 뒷받침할 근거가 부족하면 정확히 다음 문장으로 답하세요:
'제공된 논문에서 이 질문에 답할 근거를 찾지 못했습니다.'
가능하면 핵심을 먼저 말하고, 답변 끝에 근거 페이지를 괄호로 표시하세요.""",
        ),
        ("human", "질문: {question}\n\n논문 문맥:\n{context}"),
    ]
)


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
        if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
            raise RuntimeError("GOOGLE_API_KEY 또는 GEMINI_API_KEY가 필요합니다.")

        embeddings = GoogleGenerativeAIEmbeddings(
            model=os.getenv(
                "GEMINI_EMBEDDING_MODEL",
                "models/gemini-embedding-001",
            ),
            output_dimensionality=768,
        )

        if INDEX_PATH.exists():
            vector_store = Chroma(
                collection_name="paper",
                embedding_function=embeddings,
                persist_directory=str(INDEX_PATH),
            )
        else:
            pages = load_pdf_pages(PDF_PATH)
            chunks = split_pages(pages)
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name="paper",
                collection_metadata={"hnsw:space": "cosine"},
                persist_directory=str(INDEX_PATH),
            )

        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash"),
            temperature=0,
        )

        return cls(
            vector_store,
            llm,
            min_relevance=float(os.getenv("RAG_MIN_RELEVANCE", "0.35")),
        )

    def retrieve(self, question: str, top_k: int) -> list[tuple[Document, float]]:
        matches = self.vector_store.similarity_search_with_relevance_scores(
            question,
            k=top_k,
        )
        return [
            (document, float(score))
            for document, score in matches
            if score >= self.min_relevance
        ]

    def ask(self, question: str, top_k: int = 4) -> AskResponse:
        matches = self.retrieve(question, top_k)

        if not matches:
            return AskResponse(
                answer=NO_EVIDENCE_ANSWER,
                sources=[],
                grounded=False,
            )

        context = "\n\n".join(
            f"[PDF {document.metadata['page']}페이지]\n{document.page_content}"
            for document, _ in matches
        )
        chain = PROMPT | self.llm | StrOutputParser()
        answer = chain.invoke({"question": question, "context": context})

        sources = [
            Source(
                page=int(document.metadata["page"]),
                snippet=_clean_snippet(document.page_content),
                relevance=round(score, 4),
            )
            for document, score in matches
        ]

        return AskResponse(
            answer=answer,
            sources=sources,
            grounded=answer.strip() != NO_EVIDENCE_ANSWER,
        )


def _clean_snippet(text: str, max_length: int = 280) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_length:
        return compact
    return f"{compact[:max_length].rstrip()}..."
