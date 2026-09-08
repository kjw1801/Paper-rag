from backend.ingest import clean_extracted_text, load_pdf_pages, split_pages
from backend.service import PDF_PATH


def test_pdf_chunks_keep_one_based_page_numbers() -> None:
    pages = load_pdf_pages(PDF_PATH)
    chunks = split_pages(pages)

    assert len(pages) == 7
    assert chunks
    assert {chunk.metadata["page"] for chunk in chunks} == set(range(1, 8))
    assert all(chunk.metadata["source"] == "paper.pdf" for chunk in chunks)


def test_author_contact_footnotes_are_removed() -> None:
    text = (
        "논문 본문\n"
        "\x8a First Author : Mapssi, Co., Ltd., author@example.com, 정회원\n"
        "° Corresponding Author : University, contact@example.com, 정회원\n"
        "Key Words : group recommendation"
    )

    cleaned = clean_extracted_text(text)

    assert cleaned == "논문 본문\nKey Words : group recommendation"


def test_loaded_pdf_does_not_include_author_contact_footnotes() -> None:
    text = "\n".join(page.page_content for page in load_pdf_pages(PDF_PATH))

    assert "First Author" not in text
    assert "Corresponding Author" not in text
    assert "kjw1801@nate.com" not in text
