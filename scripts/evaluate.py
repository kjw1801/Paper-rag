import os

from dotenv import load_dotenv

from backend.service import RAGService

QUESTIONS = [
    "이 논문이 해결하려는 문제는 무엇인가요?",
    "그룹 추천은 어떤 세 단계로 구성되나요?",
    "사용자 간 유사도 계산에는 어떤 지표를 사용했나요?",
    "실험 데이터의 사용자 수와 거래 데이터 수는 얼마인가요?",
    "제안 방법의 F-measure는 얼마인가요?",
    "저자가 가장 좋아하는 음식은 무엇인가요?",
]


def main() -> None:
    load_dotenv()
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        raise SystemExit(".env에 GOOGLE_API_KEY를 입력한 뒤 다시 실행해 주세요.")

    service = RAGService.from_environment()
    for number, question in enumerate(QUESTIONS, start=1):
        response = service.ask(question)
        pages = ", ".join(str(source.page) for source in response.sources) or "없음"
        print(f"\n[{number}] {question}")
        print(f"답변: {response.answer}")
        print(f"근거 페이지: {pages}")


if __name__ == "__main__":
    main()
