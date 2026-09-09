"""논문 내·외 질문을 실행해 검색 점수 분포와 차단 결과를 확인한다.

사용법:
    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py --min-relevance 0.5 --json data/evaluation.json
    uv run python scripts/evaluate.py --delay 5   # 무료 티어 분당 한도 회피

Gemini 무료 티어는 모델별로 일일 요청 수 제한이 달라 GEMINI_CHAT_MODEL 설정에 좌우된다.
한도에 걸린 질문은 error로 기록하고 계속 진행한다.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from typing import Any

from dotenv import load_dotenv

from backend.service import RAGService, UpstreamRateLimited, api_key_configured

# (질문, 답변에 반드시 들어가야 할 키워드, 인용돼야 할 PDF 페이지)
IN_SCOPE: list[tuple[str, list[str], int]] = [
    ("이 논문이 해결하려는 문제는 무엇인가요?", ["그룹"], 1),
    ("그룹 추천은 어떤 세 단계로 구성되나요?", ["빈발 패턴", "유사도", "추천 목록"], 4),
    ("사용자 간 유사도 계산에는 어떤 지표를 사용했나요?", ["자카드"], 2),
    ("실험 데이터의 사용자 수와 선호도 데이터 수는 얼마인가요?", ["248", "19,105"], 5),
    ("제안 방법의 F-measure는 얼마인가요?", ["0.15435"], 5),
]

# 논문 어휘와 다른 표현으로 물어도 같은 근거를 찾는지 보는 변형 질문
PARAPHRASES: list[tuple[str, list[str], int]] = [
    # 논문은 "선호도 데이터"라고 쓰므로 "거래 데이터"는 어휘가 어긋나는 사례
    ("실험 데이터의 사용자 수와 거래 데이터 수는 얼마인가요?", ["248", "19,105"], 5),
    ("몇 명의 사용자 데이터로 실험했나요?", ["248"], 5),
    ("제안 기법의 성능 수치를 알려주세요.", ["0.15435"], 5),
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


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def run_question(
    service: RAGService,
    question: str,
    expected_grounded: bool,
    expected_keywords: list[str] | None = None,
    expected_page: int | None = None,
) -> dict[str, Any]:
    scores = [
        {"page": int(document.metadata["page"]), "score": round(score, 4)}
        for document, score in service.retrieve(question, top_k=4)
    ]
    try:
        response = service.ask(question)
    except UpstreamRateLimited as error:
        return {
            "question": question,
            "expected_grounded": expected_grounded,
            "expected_keywords": expected_keywords or [],
            "expected_page": expected_page,
            "error": str(error),
            "grounded": None,
            "classified": False,
            "missing_keywords": [],
            "page_ok": False,
            "correct": False,
            "top_score": scores[0]["score"] if scores else 0.0,
            "scores": scores,
            "answer": "",
            "cited_pages": [],
        }

    answer = _normalize(response.answer)
    missing_keywords = [
        keyword
        for keyword in (expected_keywords or [])
        if _normalize(keyword) not in answer
    ]
    page_ok = expected_page is None or expected_page in response.cited_pages
    classified = response.grounded == expected_grounded

    return {
        "question": question,
        "expected_grounded": expected_grounded,
        "expected_keywords": expected_keywords or [],
        "expected_page": expected_page,
        "grounded": response.grounded,
        "classified": classified,
        "missing_keywords": missing_keywords,
        "page_ok": page_ok,
        # 분류가 맞고, 키워드가 모두 들어 있고, 기대 페이지가 인용돼야 정답
        "correct": classified and not missing_keywords and page_ok,
        "top_score": scores[0]["score"] if scores else 0.0,
        "scores": scores,
        "answer": response.answer,
        "cited_pages": response.cited_pages,
    }


def print_result(number: int, result: dict[str, Any]) -> None:
    mark = "OK " if result["correct"] else ("ERR " if result.get("error") else "FAIL")
    scores = ", ".join(f"p{s['page']}:{s['score']:.3f}" for s in result["scores"])
    print(f"\n[{number}] {mark} {result['question']}")
    print(f"    검색 점수: {scores}")
    print(f"    grounded={result['grounded']} cited_pages={result['cited_pages']}")
    if result["missing_keywords"]:
        print(f"    누락 키워드: {result['missing_keywords']}")
    if result["expected_page"] is not None and not result["page_ok"]:
        print(f"    기대 페이지 {result['expected_page']}가 인용되지 않음")
    if result.get("error"):
        print(f"    오류: {result['error']}")
    else:
        print(f"    답변: {result['answer']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-relevance", type=float, help="RAG_MIN_RELEVANCE 대신 사용할 임계값"
    )
    parser.add_argument("--json", type=Path, help="결과를 저장할 JSON 파일 경로")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="질문 사이 대기 시간(초). 분당 한도 회피용",
    )
    parser.add_argument(
        "--no-paraphrases",
        action="store_true",
        help="표현 변형 질문을 생략해 호출 수를 줄임",
    )
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
    for number, (question, keywords, page) in enumerate(IN_SCOPE, start=1):
        result = run_question(
            service,
            question,
            expected_grounded=True,
            expected_keywords=keywords,
            expected_page=page,
        )
        results.append(result)
        print_result(number, result)
        time.sleep(args.delay)

    print("\n=== 범위 밖 질문 ===")
    for number, question in enumerate(OUT_OF_SCOPE, start=len(IN_SCOPE) + 1):
        result = run_question(service, question, expected_grounded=False)
        results.append(result)
        print_result(number, result)
        time.sleep(args.delay)

    paraphrase_results: list[dict[str, Any]] = []
    if not args.no_paraphrases:
        print("\n=== 표현 변형 질문 ===")
        start = len(IN_SCOPE) + len(OUT_OF_SCOPE) + 1
        for number, (question, keywords, page) in enumerate(PARAPHRASES, start=start):
            result = run_question(
                service,
                question,
                expected_grounded=True,
                expected_keywords=keywords,
                expected_page=page,
            )
            paraphrase_results.append(result)
            print_result(number, result)
            time.sleep(args.delay)

    in_scores = [r["top_score"] for r in results if r["expected_grounded"]]
    out_scores = [r["top_score"] for r in results if not r["expected_grounded"]]
    in_min, out_max = min(in_scores), max(out_scores)
    classified = sum(r["classified"] for r in results)
    correct = sum(r["correct"] for r in results)

    errors = sum(1 for r in results if r.get("error"))

    print("\n=== 요약 ===")
    if errors:
        print(f"호출 실패(한도 초과 등): {errors}개 — 결과는 불완전합니다")
    print(f"grounded 분류 일치: {classified}/{len(results)}")
    print(f"정답 (분류 + 키워드 + 인용 페이지): {correct}/{len(results)}")
    if paraphrase_results:
        paraphrase_correct = sum(r["correct"] for r in paraphrase_results)
        print(
            f"표현 변형 정답 (핵심 지표와 별도 집계): {paraphrase_correct}/{len(paraphrase_results)}"
        )
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
                    "errors": errors,
                    "classified": classified,
                    "correct": correct,
                    "total": len(results),
                    "suggested_threshold": suggested,
                    "results": results,
                    "paraphrase_results": paraphrase_results,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"결과 저장: {args.json}")


if __name__ == "__main__":
    main()
