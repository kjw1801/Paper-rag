"""배포 이미지에 인덱스가 빠지지 않는지 지킨다.

`.dockerignore`는 `.gitignore`와 문법이 다르다. Docker는 각 패턴에 filepath.Clean을
적용해 끝의 '/'를 떼기 때문에, 디렉터리만 제외하려고 쓴 `data/chroma/*/`가
`data/chroma/*`가 되어 같은 폴더의 chroma.sqlite3까지 빠진다. 그러면 이미지에
인덱스가 없는 채로 배포돼 /health가 document_count=0을 돌려준다.
"""

import fnmatch
import posixpath
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SEGMENT_DIR = "data/chroma/4b689855-3a23-4164-afe9-af5309d3a658"


def _docker_excludes(path: str) -> bool:
    """Docker의 .dockerignore 판정을 흉내 낸다. 마지막에 일치한 패턴이 이긴다."""
    excluded = False
    for line in (ROOT_DIR / ".dockerignore").read_text(encoding="utf-8").splitlines():
        pattern = line.strip()
        if not pattern or pattern.startswith("#"):
            continue
        negated = pattern.startswith("!")
        pattern = posixpath.normpath(pattern.removeprefix("!"))
        if fnmatch.fnmatchcase(path, pattern) or path.startswith(f"{pattern}/"):
            excluded = not negated
    return excluded


def test_index_sqlite_reaches_the_image() -> None:
    assert not _docker_excludes("data/chroma/chroma.sqlite3")


def test_regenerable_segment_directory_stays_out_of_the_image() -> None:
    assert _docker_excludes(SEGMENT_DIR)
    assert _docker_excludes(f"{SEGMENT_DIR}/data_level0.bin")


def test_dockerfile_copies_the_index() -> None:
    dockerfile = (ROOT_DIR / "Dockerfile").read_text(encoding="utf-8")

    assert "COPY data/chroma ./data/chroma" in dockerfile
