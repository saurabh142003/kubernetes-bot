FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    LANGSMITH_TRACING=false

# OpenMP runtime used by onnxruntime (FlashRank / NeMo embeddings)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so this layer is cached until the requirements change
COPY requirements-prod.txt constraints.txt ./
RUN pip install --prefer-binary -r requirements-prod.txt -c constraints.txt

COPY app/ ./app/

# Download the local models once at build time, so the first request is not slow
RUN python -c "from flashrank import Ranker; Ranker(cache_dir='/tmp/flashrank')" \
 && python -c "from app.guardrails import initialize_rails; initialize_rails()"

EXPOSE 8000
# Render sets $PORT; fall back to 8000 for local runs
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
