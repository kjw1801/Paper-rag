from functools import lru_cache
from typing import Annotated

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.models import AskRequest, AskResponse, HealthResponse
from backend.service import RAGService, api_key_configured, index_document_count

load_dotenv()

app = FastAPI(
    title="Paper RAG API",
    description="논문 근거와 PDF 페이지를 함께 반환하는 RAG API",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@lru_cache(maxsize=1)
def get_service() -> RAGService:
    try:
        return RAGService.from_environment()
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


ServiceDependency = Annotated[RAGService, Depends(get_service)]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    count = index_document_count()
    return HealthResponse(
        status="ok",
        api_key_configured=api_key_configured(),
        index_ready=count > 0,
        document_count=count,
    )


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, service: ServiceDependency) -> AskResponse:
    try:
        return service.ask(payload.question, payload.top_k)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


def run() -> None:
    uvicorn.run("backend.api:app", host="127.0.0.1", port=8000, reload=True)
