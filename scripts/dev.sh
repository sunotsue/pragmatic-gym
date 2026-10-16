#!/usr/bin/env bash
# One-command local development: API on :8000, web on :3000.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ ! -x backend/.venv/bin/python ]; then
  echo "→ creating backend venv"; (cd backend && uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[dev]")
fi
if [ ! -d apps/web/node_modules ]; then
  echo "→ installing web deps"; (cd apps/web && npm install)
fi
if [ ! -f data/pools/train_pool.jsonl ]; then
  echo "→ building candidate pool and evaluation sets"; backend/.venv/bin/python -c "import sys; sys.path.insert(0,'backend'); from pgym.selection.pool import build_all; print(build_all())"
fi

trap 'kill 0' EXIT
(cd backend && .venv/bin/uvicorn pgym.api.main:app --port 8000 --log-level warning) &
(cd apps/web && npx next dev -p 3000) &
echo "Pragmatic Gym → http://localhost:3000   (API http://localhost:8000/docs)"
wait
