# Paper RAG

연구 논문 **「협업 필터링과 빈발 패턴을 이용한 개인화된 그룹 추천」**을 대상으로 만든 근거 기반 질의응답 데모입니다. 질문과 의미적으로 가까운 논문 청크를 Chroma에서 검색하고, Gemini가 검색 문맥만으로 답변하도록 제한합니다. 응답에는 근거 문장과 PDF 페이지가 포함됩니다.

## 바로 보기

- **공개 데모:** [paper.woojulab.com](https://paper.woojulab.com/)
- **논문 원문:** [웹 PDF](https://paper.woojulab.com/paper.pdf) · [KCI](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002130830) · [DBpia](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE06747562)
- **소스 코드:** [github.com/kjw1801/Paper-rag](https://github.com/kjw1801/Paper-rag)

## 대상 논문

- **저자:** 김정우, 박광현
- **게재:** 한국통신학회논문지, 2016년 7월, 41권 7호, 768-774쪽
- **DOI:** [10.7840/kics.2016.41.7.768](https://doi.org/10.7840/kics.2016.41.7.768)
- **연구 내용:** 단일 상품 추천의 한계를 보완하기 위해 협업 필터링, 연관 규칙과 빈발 패턴을 결합해 패션처럼 서로 연관된 상품 그룹을 개인화하여 추천하는 방법을 제안합니다.
- **실험 결과:** 의류 전자상거래 사용자 248명의 선호도 데이터 19,105개로 평가했으며, 제안한 그룹 추천의 평균 F-measure는 `0.15435`로 기존 협업 필터링의 `0.0977`보다 높았습니다.

## 구현 내용

- `pypdf`로 7페이지 PDF 텍스트 추출
- 900자 청크, 150자 overlap으로 분할
- 원본 PDF 페이지 번호를 각 청크의 metadata로 보존
- Gemini `gemini-embedding-001` 임베딩
- 서버 없이 파일로 저장되는 로컬 Chroma 인덱스
- LangChain prompt/model/output parser 파이프라인
- Gemini JSON 모드로 `answer`, `has_evidence`, `cited_pages`를 구조화해 받고, `grounded` 판정을 답변 문자열과 분리
- JSON 파싱 실패나 검색된 페이지를 인용하지 않은 답변은 근거 없음으로 차단 (fail-closed)
- 유사도 임계값(1차)과 모델의 `has_evidence`(2차)를 함께 써서 범위 밖 질문 차단
- 청크 안에서 질문과 가장 겹치는 문장을 근거 문장으로 반환하고, UI에서 PDF 해당 페이지로 링크
- FastAPI `POST /ask`
- React 질문·답변·출처 확인 화면
- Cloudflare Turnstile 서버 검증과 전역 요청량·동시 호출 제한
- 무료 API 한도와 일시 장애를 구분하는 HTTP 상태·오류 코드 표시

## 구조

```text
PDF
 └─ 페이지별 텍스트 추출
     └─ metadata: page, source
         └─ RecursiveCharacterTextSplitter
             └─ Gemini Embeddings
                 └─ Chroma local index
                     └─ top-k retrieval
                         └─ Gemini grounded answer
                             └─ answer + snippet + PDF page
```

```text
backend/
  api.py        FastAPI 엔드포인트와 CORS 설정
  guard.py      전역 분당·일일 요청량과 동시 호출 제한
  ingest.py     PDF 페이지 추출과 청크 분할
  models.py     요청·응답 스키마
  service.py    인덱스 생성·검증, Chroma 검색, Gemini 구조화 답변
  turnstile.py  Cloudflare Turnstile 토큰 검증
data/
  paper.pdf     검색 대상 논문
scripts/
  build_index.py
  evaluate.py   논문 내 5개·범위 밖 6개 질문으로 점수 분포와 차단 결과 확인
tests/
web/            React UI
```

## 설치

Python 3.12와 `uv`를 사용합니다.

```bash
uv sync
cp .env.example .env
```

`.env`에 본인의 Gemini API 키를 입력합니다. API 키는 React 코드가 아니라 FastAPI 서버에서만 읽으며 `.env`는 Git에서 제외됩니다. 공개 배포는 Turnstile을 기본 필수로 처리하며, 로컬 개발에서만 `TURNSTILE_REQUIRED=0`으로 명시적으로 끕니다.

```dotenv
GOOGLE_API_KEY=your_key_here
GEMINI_CHAT_MODEL=gemini-3.5-flash-lite
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001
RAG_MIN_RELEVANCE=0.60
TURNSTILE_REQUIRED=0
```

## 실행

최초 한 번 Chroma 인덱스를 생성합니다. 인덱스가 없거나 비어 있으면 서버가 `/ask`에서 503으로 알려줍니다. 청크 설정을 바꿨다면 `--rebuild`로 다시 만듭니다.

```bash
uv run python scripts/build_index.py
```

```bash
uv run python scripts/build_index.py --rebuild
```

백엔드를 실행합니다.

```bash
uv run python main.py
```

다른 터미널에서 React 화면을 실행합니다.

```bash
cd web
npm install
npm run dev
```

프런트 환경변수는 `web/.env.example`을 참고합니다.

- React: http://localhost:3000
- FastAPI 문서: http://127.0.0.1:8000/docs
- 상태 확인: http://127.0.0.1:8000/health (`document_count`로 인덱스 청크 수 확인)

## API

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"제안 방법의 F-measure는 얼마인가요?","top_k":4}'
```

응답 예시:

```json
{
  "answer": "제안한 그룹 추천 방법의 F-measure는 0.15435입니다. (PDF 5페이지)",
  "sources": [
    {
      "page": 5,
      "snippet": "표 1은 추천 성능 평가 결과이다. 그룹 추천의 F-measure는 0.15435이다.",
      "relevance": 0.82,
      "cited": true
    }
  ],
  "grounded": true,
  "cited_pages": [5]
}
```

## 검증 질문

`uv run python scripts/evaluate.py`를 실행하면 논문 내 질문 5개와 범위 밖 질문 6개를 실제 검색과 Gemini 호출로 검사하고, 질문별 검색 점수·grounded 판정·인용 페이지와 함께 임계값 제안을 출력합니다.

```bash
uv run python scripts/evaluate.py --json data/evaluation.json
```

논문 내 질문:

1. 이 논문이 해결하려는 문제는 무엇인가요?
2. 그룹 추천은 어떤 세 단계로 구성되나요?
3. 사용자 간 유사도 계산에는 어떤 지표를 사용했나요?
4. 실험 데이터의 사용자 수와 선호도 데이터 수는 얼마인가요?
5. 제안 방법의 F-measure는 얼마인가요?

논문 내 질문은 `grounded` 분류가 맞고, 기대 키워드(예: `자카드`, `0.15435`)가 답변에 들어 있고, 기대 페이지가 `cited_pages`에 포함돼야 정답으로 셉니다. 요약에는 분류 일치율과 정답률을 따로 표시합니다.

범위 밖 질문:

6. 저자가 가장 좋아하는 음식은 무엇인가요?
7. 내일 서울 날씨는 어떤가요?
8. 파이썬에서 리스트를 정렬하는 방법은 무엇인가요?
9. 삼성전자의 올해 주가 전망은 어떤가요?
10. 이 논문의 저자가 태어난 해는 언제인가요? - 논문 어휘를 쓰지만 본문에 없는 정보
11. 넷플릭스에서 지금 가장 인기 있는 드라마는 무엇인가요? - 추천 도메인과 겹치는 무관한 질문

표현 변형 질문 (논문 어휘와 다른 말로 물어도 같은 근거를 찾는지 확인, `--no-paraphrases`로 생략 가능):

12. 실험 데이터의 사용자 수와 거래 데이터 수는 얼마인가요? - 논문은 "선호도 데이터"라고 씀
13. 몇 명의 사용자 데이터로 실험했나요?
14. 제안 기법의 성능 수치를 알려주세요.

임계값 `RAG_MIN_RELEVANCE`는 논문 내 질문의 최고 점수 최솟값과 범위 밖 질문의 최고 점수 최댓값 사이에 둡니다. 두 구간이 겹치면 임계값은 논문 내 질문을 놓치지 않는 값으로 두고 모델의 `has_evidence` 판정이 차단을 담당합니다.

Gemini 무료 티어의 요청 한도는 모델과 계정 상태에 따라 달라질 수 있습니다. 평가 14개 질문은 호출량을 많이 사용하므로 반복 실행을 피하고, 한도에 걸린 질문은 `ERR`로 기록합니다. `--delay`는 분당 한도를 완화할 뿐 일일 한도는 해결하지 못합니다. 서버는 검색 임베딩과 답변 생성 어느 단계에서든 한도에 걸리면 `AI_RATE_LIMITED`와 HTTP 429를 반환합니다.

단위 테스트는 외부 API를 호출하지 않고 페이지 metadata 보존, 근거 문장 선택, 구조화 답변 파싱과 grounded 판정, 인용 페이지 강제, 빈 인덱스 감지, 안전한 재생성, API 응답 구조를 검증합니다.

```bash
uv run ruff check .
uv run pyright
uv run pytest
```

## 배포

로컬 기본값은 개발용입니다. 배포할 때는 아래 환경변수를 추가로 설정합니다.

| 변수 | 용도 |
|---|---|
| `CORS_ALLOWED_ORIGINS` | React 배포 주소를 쉼표로 구분해 추가 (기본값은 localhost:3000만 허용) |
| `RAG_INDEX_DIR` | Chroma 인덱스 경로. `data/chroma/`는 Git에 포함되지 않으므로 영구 디스크 경로를 지정 |
| `RAG_BUILD_INDEX_ON_STARTUP=1` | 인덱스가 없으면 서버 시작 시 한 번 생성 (임베딩 호출 발생). 파일 잠금으로 중복 생성을 막지만 worker 1개로 시작하는 것을 권장 |
| `RAG_RATE_LIMIT_PER_MINUTE` | 인스턴스 전체가 1분 동안 처리할 질문 수 (기본 15) |
| `RAG_DAILY_REQUEST_LIMIT` | 인스턴스가 UTC 하루 동안 처리할 질문 수 (기본 500) |
| `RAG_MAX_CONCURRENT_REQUESTS` | 동시에 실행할 Gemini 요청 수 (기본 2) |
| `TURNSTILE_SECRET_KEY` | Cloudflare Turnstile 비밀 키. 백엔드에서만 관리 |
| `TURNSTILE_REQUIRED=1` | 공개 배포에서 토큰 검증을 필수화 |
| `TURNSTILE_EXPECTED_HOSTNAMES` | `paper.woojulab.com` 등 허용 호스트 |
| `TURNSTILE_EXPECTED_ACTION` | 프런트와 동일한 `paper_ask` 사용 |
| `HOST=0.0.0.0`, `PORT` | 배포 플랫폼이 요구하는 바인딩 주소와 포트 |
| `RAG_RELOAD=0` | 자동 리로드 끄기 |

React 쪽은 `NEXT_PUBLIC_API_URL`에 백엔드 배포 주소를, `NEXT_PUBLIC_TURNSTILE_SITE_KEY`에 공개 사이트 키를 넣습니다. 인덱스 재생성은 임시 디렉터리에 만든 뒤 교체하므로 실패해도 기존 인덱스가 유지됩니다.

### 공개 데모 보호

`/ask`는 Cloudflare Turnstile 검증을 먼저 통과해야 하며, 인스턴스 전체의 분당·일일 요청 수와 동시 모델 호출 수를 제한합니다. 호출자가 조작할 수 있는 `X-Forwarded-For` 값에는 의존하지 않습니다. 제한을 넘으면 `Retry-After`와 `DEMO_RATE_LIMITED` 코드가 포함된 HTTP 429를 반환합니다. 질문 본문은 최대 500자로 제한합니다.

메모리 기반 일일 카운터는 Cloud Run 인스턴스가 재시작되면 초기화되는 보조 장치입니다. 실제 비용 상한은 Google Cloud/Gemini 프로젝트의 결제 설정과 API 할당량으로 별도 관리해야 합니다.

## 한계

- 7페이지 단일 논문에 맞춘 데모이며 여러 문서 업로드 기능은 없습니다.
- PDF의 표와 수식은 텍스트 추출 결과에 따라 순서가 달라질 수 있습니다.
- 평가 질문에서 논문 내 최고 점수는 `0.683~0.774`, 범위 밖 질문은 `0.502~0.667`이었습니다. 현재 임계값 `0.60`은 관련 질문의 재현율을 보존하는 1차 필터이며, 경계에 걸친 질문은 모델의 `has_evidence`가 2차로 차단합니다.
- 근거 문장은 질문과 글자 2-gram이 가장 많이 겹치는 문장을 고르는 단순 방식입니다.
- 답변 품질은 Gemini 모델과 검색된 청크에 영향을 받습니다.
- 개인 프로젝트이며 무료 AI API 플랜으로 운영하므로 사용량에 따라 답변이 느리거나 일시 중단될 수 있습니다. 화면의 HTTP 상태와 오류 코드로 한도 소진과 일시 장애를 구분합니다.
