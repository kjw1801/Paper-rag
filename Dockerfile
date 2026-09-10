FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY backend ./backend
COPY main.py ./main.py
COPY data/paper.pdf ./data/paper.pdf
# 이미지에 인덱스를 넣어 콜드 스타트마다 임베딩을 다시 만들지 않는다
COPY data/chroma ./data/chroma

CMD ["sh", "-c", "uvicorn backend.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
