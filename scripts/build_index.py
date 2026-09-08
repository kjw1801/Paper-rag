import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from backend.service import INDEX_PATH, build_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="논문 PDF를 Chroma 인덱스로 저장합니다."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="기존 인덱스를 지우고 다시 생성합니다 (청크 설정을 바꿨을 때 사용).",
    )
    args = parser.parse_args()

    load_dotenv()
    count = build_index(rebuild=args.rebuild)
    print(f"Chroma 인덱스 준비 완료: {INDEX_PATH} (청크 {count}개)")


if __name__ == "__main__":
    main()
