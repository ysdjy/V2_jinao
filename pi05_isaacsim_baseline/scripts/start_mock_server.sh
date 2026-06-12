#!/usr/bin/env bash
# Start the mock policy server (stdlib only -> any python). Backgrounds it and
# writes the PID to logs/mock_server.pid. Use stop_server.sh to stop it.
set -u
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ"
PORT="${1:-8008}"
PY="${PYBIN:-python3}"
LOG="logs/mock_server_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs
nohup "$PY" adapters/policy_server/server.py --backend mock --port "$PORT" > "$LOG" 2>&1 &
echo $! > logs/mock_server.pid
echo "mock server pid $(cat logs/mock_server.pid) port $PORT log $LOG"
# wait for health
for i in $(seq 1 20); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then echo "healthy"; exit 0; fi
  sleep 0.5
done
echo "WARN: server did not become healthy in time"; exit 1
