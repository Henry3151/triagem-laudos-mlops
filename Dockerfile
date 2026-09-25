# syntax=docker/dockerfile:1

# ---------- base com uv ----------
FROM python:3.12-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# ---------- treino: treina, exporta ONNX e promove ----------
FROM base AS treino
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --extra treino --no-install-project
COPY src ./src
COPY data/raw ./data/raw
RUN uv sync --locked --no-dev --extra treino \
    && uv run --no-sync python -m triagem.treinar \
        --dados data/raw/laudos_sinteticos.csv --models-dir /build/models --trabalho /tmp/trabalho

# ---------- dependências de serving (sem pandas/skl2onnx) ----------
FROM base AS deps
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

# ---------- runtime enxuto ----------
FROM python:3.12-slim AS runtime
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=deps /app/.venv /app/.venv
COPY --from=treino /build/models/producao /app/models/producao
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models/producao \
    MODEL_BACKEND=onnx \
    PORT=8000
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f\"http://127.0.0.1:{os.environ.get('PORT', '8000')}/health\", timeout=2)"]
CMD ["sh", "-c", "exec uvicorn triagem.api.app:app --host 0.0.0.0 --port ${PORT}"]
