import langchain
import langchain_openai


def test_langchain_packages_are_available() -> None:
    assert langchain is not None
    assert langchain_openai is not None
