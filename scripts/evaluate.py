"""논문 내·외 질문을 실행해 검색 점수 분포와 차단 결과를 확인한다.

사용법:
    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py --min-relevance 0.5 --json data/evaluation.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from typing import Any

from dotenv import load_dotenv

from backend.service import RAGService, api_key_configured

IN_SCOPE = [
    "이 논문이 해결하려는 문제는 무엇인가요?",
    "그룹 추천은 어떤 세 단계로 구성되나요?",
    "사용자 간 유사도 계산에는 어떤 지표를 사용했나요?",
    "실험 데이터의 사용자 수와 거래 데이터 수는 얼마인가요?",
    "제안 방법의 F-measure는 얼마인가요?",
]

OUT_OF_SCOPE = [
    "저자가 가장 좋아하는 음식은 무엇인가요?",
    "내일 서울 날씨는 어떤가요?",
    "파이썬에서 리스트를 정렬하는 방법은 무엇인가요?",
    "삼성전자의 올해 주가 전망은 어떤가요?",
    # 논문 어휘를 쓰지만 본문에 없는 정보를 묻는 어려운 사례
    "이 논문의 저자가 태어난 해는 언제인가요?",
    # 추천 도메인과 겹치지만 논문과 무관한 어려운 사례
    "넷플릭스에서 지금 가장 인기 있는 드라마는 무엇인가요?",
]


def run_question(
    service: RAGService, question: str, expected_grounded: bool
) -> dict[str, Any]:
    scores = [
        {"page": int(document.metadata["page"]), "score": round(score, 4)}
        for document, score in service.retrieve(question, top_k=4)
    ]
    response = service.ask(question)
    return {
        "question": question,
        "expected_grounded": expected_grounded,
        "grounded": response.grounded,
        "correct": response.grounded == expected_grounded,
        "top_score": scores[0]["score"] if scores else 0.0,
        "scores": scores,
        "answer": response.answer,
        "cited_pages": response.cited_pages,
    }


def print_result(number: int, result: dict[str, Any]) -> None:
    mark = "OK " if result["correct"] else "FAIL"
    scores = ", ".join(f"p{s['page']}:{s['score']:.3f}" for s in result["scores"])
    print(f"\n[{number}] {mark} {result['question']}")
    print(f"    검색 점수: {scores}")
    print(f"    grounded={result['grounded']} cited_pages={result['cited_pages']}")
    print(f"    답변: {result['answer']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-relevance", type=float, help="RAG_MIN_RELEVANCE 대신 사용할 임계값"
    )
    parser.add_argument("--json", type=Path, help="결과를 저장할 JSON 파일 경로")
    args = parser.parse_args()

    load_dotenv()
    if not api_key_configured():
        raise SystemExit(".env에 GOOGLE_API_KEY를 입력한 뒤 다시 실행해 주세요.")

    service = RAGService.from_environment()
    if args.min_relevance is not None:
        service.min_relevance = args.min_relevance
    print(f"임계값 RAG_MIN_RELEVANCE={service.min_relevance}")

    results: list[dict[str, Any]] = []
    print("\n=== 논문 내 질문 ===")
    for number, question in enumerate(IN_SCOPE, start=1):
        result = run_question(service, question, expected_grounded=True)
        results.append(result)
        print_result(number, result)

    print("\n=== 범위 밖 질문 ===")
    for number, question in enumerate(OUT_OF_SCOPE, start=len(IN_SCOPE) + 1):
        result = run_question(service, question, expected_grounded=False)
        results.append(result)
        print_result(number, result)

    in_scores = [r["top_score"] for r in results if r["expected_grounded"]]
    out_scores = [r["top_score"] for r in results if not r["expected_grounded"]]
    in_min, out_max = min(in_scores), max(out_scores)
    correct = sum(r["correct"] for r in results)

    print("\n=== 요약 ===")
    print(f"정답: {correct}/{len(results)}")
    print(f"논문 내 질문 최고 점수 범위: {min(in_scores):.3f} ~ {max(in_scores):.3f}")
    print(f"범위 밖 질문 최고 점수 범위: {min(out_scores):.3f} ~ {out_max:.3f}")
    if in_min > out_max:
        suggested = round((in_min + out_max) / 2, 3)
        print(
            f"점수만으로 분리 가능. 임계값 제안: {suggested} (여유 {in_min - out_max:.3f})"
        )
    else:
        suggested = None
        print(
            "점수만으로는 분리되지 않음. 임계값은 논문 내 질문을 놓치지 않는 값으로 두고 "
            "has_evidence 판정에 의존해야 함."
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "min_relevance": service.min_relevance,
                    "correct": correct,
                    "total": len(results),
                    "suggested_threshold": suggested,
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"결과 저장: {args.json}")


if __name__ == "__main__":
    main()
