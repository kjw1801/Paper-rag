# Paper RAG

본인 논문인 **「협업 필터링과 빈발 패턴을 이용한 개인화된 그룹 추천」**을 대상으로 만든 근거 기반 질의응답 데모입니다. 질문과 의미적으로 가까운 논문 청크를 Chroma에서 검색하고, Gemini가 검색 문맥만으로 답변하도록 제한합니다. 응답에는 근거 문장과 PDF 페이지가 포함됩니다.

## 구현 내용

- `pypdf`로 7페이지 PDF 텍스트 추출
- 900자 청크, 150자 overlap으로 분할
- 원본 PDF 페이지 번호를 각 청크의 metadata로 보존
- Gemini `gemini-embedding-001` 임베딩
- 서버 없이 파일로 저장되는 로컬 Chroma 인덱스
- LangChain prompt/model/output parser 파이프라인
- 유사도 임계값과 제한 프롬프트를 이용한 범위 밖 질문 처리
- FastAPI `POST /ask`
- React 질문·답변·출처 확인 화면

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
  ingest.py     PDF 페이지 추출과 청크 분할
  models.py     요청·응답 스키마
  service.py    임베딩, FAISS 검색, Gemini 답변
data/
  paper.pdf     검색 대상 논문
scripts/
  build_index.py
  evaluate.py   논문 내·외 질문 6개 검증
tests/
web/            React UI
```

## 설치

Python 3.12와 `uv`를 사용합니다.

```bash
uv sync
cp .env.example .env
```

`.env`에 본인의 Gemini API 키를 입력합니다. API 키는 React 코드가 아니라 FastAPI 서버에서만 읽으며 `.env`는 Git에서 제외됩니다.

```dotenv
GOOGLE_API_KEY=your_key_here
GEMINI_CHAT_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001
RAG_MIN_RELEVANCE=0.35
```

## 실행

최초 한 번 Chroma 인덱스를 생성합니다.

```bash
uv run python scripts/build_index.py
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

- React: http://localhost:3000
- FastAPI 문서: http://127.0.0.1:8000/docs
- 상태 확인: http://127.0.0.1:8000/health

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
      "snippet": "...표 1 추천 성능 평가...",
      "relevance": 0.82
    }
  ],
  "grounded": true
}
```

## 검증 질문

`uv run python scripts/evaluate.py`를 실행하면 다음 질문을 실제 검색과 Gemini 호출로 검사합니다.

1. 이 논문이 해결하려는 문제는 무엇인가요?
2. 그룹 추천은 어떤 세 단계로 구성되나요?
3. 사용자 간 유사도 계산에는 어떤 지표를 사용했나요?
4. 실험 데이터의 사용자 수와 거래 데이터 수는 얼마인가요?
5. 제안 방법의 F-measure는 얼마인가요?
6. 저자가 가장 좋아하는 음식은 무엇인가요? - 문서에 없는 질문

단위 테스트는 외부 API를 호출하지 않고 페이지 metadata 보존, 근거 반환, 유사도 임계값, API 응답 구조를 검증합니다.

```bash
uv run ruff check .
uv run pyright
uv run pytest
```

## 한계

- 7페이지 단일 논문에 맞춘 데모이며 여러 문서 업로드 기능은 없습니다.
- PDF의 표와 수식은 텍스트 추출 결과에 따라 순서가 달라질 수 있습니다.
- 유사도 임계값 `0.35`는 초기값입니다. 실제 평가 질문 결과에 맞춰 조정해야 합니다.
- 답변 품질은 Gemini 모델과 검색된 청크에 영향을 받습니다.
