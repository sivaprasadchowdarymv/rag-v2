.PHONY: install dev test lint security audit run api bench backup
install: ; pip install -r requirements.lock
dev: ; pip install -r requirements-dev.txt
test: ; python -m pytest -q
lint: ; ruff check .
security: ; bandit -c pyproject.toml -r . -ll
audit: ; pip-audit -r requirements.lock
run: ; streamlit run app.py
api: ; uvicorn api.main:app --host 127.0.0.1 --port 8000
bench: ; python -m evaluation.benchmark
backup: ; python -m scripts.backup create --dest backups
