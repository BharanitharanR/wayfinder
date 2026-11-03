#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=".venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="python3"
if [[ ! -f .env ]]; then echo "Copy .env.example to .env and set MODEL_API_KEY"; exit 1; fi
if ! "$PYTHON" -c "import uvicorn, multipart" >/dev/null 2>&1; then "$PYTHON" -m pip install -r requirements.txt; fi
if ! "$PYTHON" -c "from pymongo import MongoClient; MongoClient('mongodb://localhost:27017',serverSelectionTimeoutMS=1200).admin.command('ping')" >/dev/null 2>&1; then echo "MongoDB is not running at localhost:27017"; exit 1; fi
( sleep 2; open "http://127.0.0.1:${WAYFINDER_PORT:-8765}" 2>/dev/null || true ) &
exec "$PYTHON" -m mesh.orchestrator.wayfinder_web
