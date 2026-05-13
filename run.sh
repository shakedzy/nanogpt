#!/usr/bin/env bash
# Boot the FastAPI backend and Vite frontend together.
# Ctrl+C stops both.
set -euo pipefail

cd "$(dirname "$0")"

cleanup() {
  trap - EXIT INT TERM
  [[ -n "${BACK_PID:-}" ]] && kill "$BACK_PID" 2>/dev/null || true
  [[ -n "${FRONT_PID:-}" ]] && kill "$FRONT_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [[ ! -d frontend/node_modules ]]; then
  echo "[run] installing frontend deps..."
  (cd frontend && npm install)
fi

echo "[run] starting backend  -> http://127.0.0.1:8000"
uv run uvicorn server.main:app --reload --port 8000 --host 127.0.0.1 &
BACK_PID=$!

echo "[run] starting frontend -> http://localhost:5173"
(cd frontend && npm run dev) &
FRONT_PID=$!

while kill -0 "$BACK_PID" 2>/dev/null && kill -0 "$FRONT_PID" 2>/dev/null; do
  sleep 1
done
