# RAG Ultimate 2.0 — one image for the Streamlit UI (default) and the REST API (command override).
# Multi-stage: build wheels in a builder, ship a slim runtime with tesseract, running as a non-root user.
ARG PYTHON_IMAGE=python:3.12-slim

FROM ${PYTHON_IMAGE} AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY requirements.lock requirements-optional.txt ./
RUN pip wheel --wheel-dir /wheels -r requirements.lock \
 && pip wheel --wheel-dir /wheels -r requirements-optional.txt

FROM ${PYTHON_IMAGE} AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/app/data HF_HOME=/app/data/hf FASTEMBED_CACHE_PATH=/app/data/models LOG_FORMAT=json
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl tesseract-ocr tesseract-ocr-eng postgresql-client \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links /wheels /wheels/* && rm -rf /wheels
COPY --chown=app:app . .
RUN mkdir -p /app/data && chown -R app:app /app/data
USER app
EXPOSE 8501 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || curl -fsS http://localhost:8000/healthz || exit 1
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
