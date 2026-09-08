from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def load_pdf_pages(pdf_path: Path) -> list[Document]:
    reader = PdfReader(str(pdf_path))
    pages = [
        Document(
            page_content=pdf_page.extract_text() or "",
            metadata={"page": page_number, "source": pdf_path.name},
        )
        for page_number, pdf_page in enumerate(reader.pages, start=1)
    ]

    return pages


def split_pages(
    pages: list[Document],
    *,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    for chunk_number, chunk in enumerate(chunks):
        chunk.metadata["chunk"] = chunk_number

    return chunks
