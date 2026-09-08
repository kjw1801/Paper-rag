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

CMD ["sh", "-c", "uvicorn backend.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
