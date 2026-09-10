from pathlib import Path

import chromadb
import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings

from backend.service import COLLECTION_NAME, build_index, index_document_count


def test_missing_index_counts_zero_without_creating_directory(tmp_path: Path) -> None:
    index_path = tmp_path / "chroma"

    assert index_document_count(index_path) == 0
    assert not index_path.exists()


def test_empty_directory_counts_zero(tmp_path: Path) -> None:
    index_path = tmp_path / "chroma"
    index_path.mkdir()

    assert index_document_count(index_path) == 0


def test_persisted_index_reports_chunk_count(tmp_path: Path) -> None:
    index_path = tmp_path / "chroma"
    Chroma.from_documents(
        [
            Document(page_content="a", metadata={"page": 1}),
            Document(page_content="b", metadata={"page": 2}),
        ],
        embedding=FakeEmbeddings(size=8),
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},
        persist_directory=str(index_path),
    )

    assert index_document_count(index_path) == 2


class FailingEmbeddings(FakeEmbeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("임베딩 API 실패")


def test_failed_rebuild_keeps_existing_index(tmp_path: Path) -> None:
    index_path = tmp_path / "chroma"

    first = build_index(embeddings=FakeEmbeddings(size=8), index_path=index_path)
    assert first == index_document_count(index_path) > 0

    with pytest.raises(RuntimeError, match="임베딩 API 실패"):
        build_index(
            rebuild=True, embeddings=FailingEmbeddings(size=8), index_path=index_path
        )

    assert index_document_count(index_path) == first
    assert sorted(p.name for p in tmp_path.iterdir() if p.suffix != ".lock") == [
        "chroma"
    ]


def test_rebuild_replaces_index_and_cleans_up(tmp_path: Path) -> None:
    index_path = tmp_path / "chroma"
    build_index(embeddings=FakeEmbeddings(size=8), index_path=index_path)

    count = build_index(
        rebuild=True, embeddings=FakeEmbeddings(size=8), index_path=index_path
    )

    assert index_document_count(index_path) == count > 0
    assert sorted(p.name for p in tmp_path.iterdir() if p.suffix != ".lock") == [
        "chroma"
    ]


def test_repository_index_matches_the_current_pdf_chunks() -> None:
    """이미지에 넣는 인덱스가 PDF 최신 청크와 본문·페이지까지 같아야 한다."""
    from backend.ingest import load_pdf_pages, split_pages
    from backend.service import COLLECTION_NAME, INDEX_PATH, PDF_PATH

    chunks = split_pages(load_pdf_pages(PDF_PATH))
    assert index_document_count(INDEX_PATH) == len(chunks) == 21

    client = chromadb.PersistentClient(path=str(INDEX_PATH))
    stored = client.get_collection(COLLECTION_NAME).get(
        include=["documents", "metadatas"]
    )
    # Chroma가 id를 직접 만들므로 본문과 페이지 쌍으로 대조한다
    saved = sorted(
        (document, metadata["page"])
        for document, metadata in zip(
            stored["documents"] or [], stored["metadatas"] or [], strict=True
        )
    )
    expected = sorted((chunk.page_content, chunk.metadata["page"]) for chunk in chunks)

    assert saved == expected
