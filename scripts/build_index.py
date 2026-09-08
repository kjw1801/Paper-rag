from dotenv import load_dotenv

from backend.service import INDEX_PATH, RAGService


def main() -> None:
    load_dotenv()
    RAGService.from_environment()
    print(f"Chroma 인덱스를 생성했습니다: {INDEX_PATH}")


if __name__ == "__main__":
    main()
