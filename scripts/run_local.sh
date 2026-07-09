#!/usr/bin/env bash
set -euo pipefail

python -m pip install -r requirements.txt
uvicorn src.app:app --host "${API_HOST:-0.0.0.0}" --port "${API_PORT:-8000}"
