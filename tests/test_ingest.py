from backend.ingest import load_pdf_pages, split_pages
from backend.service import PDF_PATH


def test_pdf_chunks_keep_one_based_page_numbers() -> None:
    pages = load_pdf_pages(PDF_PATH)
    chunks = split_pages(pages)

    assert len(pages) == 7
    assert chunks
    assert {chunk.metadata["page"] for chunk in chunks} == set(range(1, 8))
    assert all(chunk.metadata["source"] == "paper.pdf" for chunk in chunks)
