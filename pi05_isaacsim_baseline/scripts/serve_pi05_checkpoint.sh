#!/usr/bin/env bash
# Serve a trained pi0.5 checkpoint through the HTTP policy server (port 8008).
# Falls back to the mock backend if the OpenPI venv / checkpoint is unavailable.
#
# Usage:
#   bash scripts/serve_pi05_checkpoint.sh \
#       --config pi05_isaaclab_franka \
#       --ckpt policies/checkpoints/pi05_isaaclab_XXXX/pi05_isaaclab_franka/isaaclab_XXXX/10 \
#       [--port 8008]
set -u
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ"
VENV_PY="$PROJ/.venv_openpi/bin/python"
PORT=8008
CONFIG=""
CKPT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2;;
    --ckpt) CKPT="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    *) shift;;
  esac
done
LOG="logs/pi05_server_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

if [ -x "$VENV_PY" ] && [ -n "$CONFIG" ] && [ -n "$CKPT" ] && [ -e "$CKPT" ]; then
  echo "[serve] real pi0.5: config=$CONFIG ckpt=$CKPT port=$PORT" | tee "$LOG"
  nohup "$VENV_PY" adapters/policy_server/server.py \
      --backend openpi --config "$CONFIG" --ckpt "$CKPT" --port "$PORT" >> "$LOG" 2>&1 &
  echo $! > logs/pi05_server.pid
else
  echo "[serve] checkpoint/venv unavailable -> mock backend on port $PORT" | tee "$LOG"
  nohup python3 adapters/policy_server/server.py --backend mock --port "$PORT" >> "$LOG" 2>&1 &
  echo $! > logs/pi05_server.pid
fi
echo "pid $(cat logs/pi05_server.pid) log $LOG"
for i in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "healthy:"; curl -fsS "http://127.0.0.1:$PORT/health"; echo; exit 0
  fi
  sleep 1
done
echo "WARN: server not healthy in time (model load can be slow); check $LOG"; exit 1
