#!/bin/sh
# Start script for the Hugging Face Docker Space (see Dockerfile.spaces).
set -e
# Persistent Storage Bucket mounted at /data? Keep everything there.
if [ -z "$DATA_DIR" ] && [ -d /data ] && [ -w /data ]; then
  export DATA_DIR=/data
fi
DATA_DIR="${DATA_DIR:-$HOME/app/data}"
export DATA_DIR
export HF_HOME="${HF_HOME:-$DATA_DIR/hf}" FASTEMBED_CACHE_PATH="${FASTEMBED_CACHE_PATH:-$DATA_DIR/models}"
mkdir -p "$DATA_DIR"
# No Ollama on Spaces: answer with the cloud chains unless told otherwise.
export LLM_MODE="${LLM_MODE:-cloud}"
echo "RAG Ultimate on Hugging Face Spaces: DATA_DIR=$DATA_DIR LLM_MODE=$LLM_MODE"
# Hosted PostgreSQL reachable from the Space? (Spaces may block outbound ports other than 80/443/8080.)
# If not, start on SQLite with a loud warning instead of crash-looping. DB_FALLBACK_SQLITE=false disables this.
if [ -n "$DATABASE_URL" ] && [ "${DB_FALLBACK_SQLITE:-true}" = "true" ]; then
  if ! python deploy/hf-space/db_probe.py; then
    echo "WARNING: DATABASE_URL is not reachable from this Space -> using SQLite in $DATA_DIR for now."
    echo "WARNING: accounts/chats will NOT be in your PostgreSQL. See docs/HF_SPACES_DEPLOY.md (Troubleshooting)."
    unset DATABASE_URL
  fi
fi
# Create / upgrade database tables (SQLite or PostgreSQL). A failure is logged, not fatal:
# the app shows the problem on its status page instead of the Space crash-looping.
python -m database.migrations || echo "WARNING: database migrations failed (check DATABASE_URL)"
exec streamlit run app.py \
  --server.address=0.0.0.0 --server.port="${PORT:-7860}" --server.headless=true \
  --server.enableXsrfProtection=false --server.enableCORS=false \
  --server.maxUploadSize="${MAX_UPLOAD_MB:-50}"
