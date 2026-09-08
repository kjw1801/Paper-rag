from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings

from backend.service import COLLECTION_NAME, index_document_count


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
