#!/usr/bin/env bash
# STAGE 4 — Serve (inference): start the HTTP policy server on $PORT.
# Defaults to the newest trained checkpoint; falls back to the mock backend if none.
#   bash WORKFLOW/4_serve.sh                       # newest checkpoint (or mock)
#   bash WORKFLOW/4_serve.sh --ckpt <dir> --port 8008
#   bash WORKFLOW/4_serve.sh --mock                # force mock backend
# Stop with: bash WORKFLOW/stop.sh
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

CK="$(latest_ckpt)"; P="$PORT"; FORCE_MOCK=0
while [ $# -gt 0 ]; do case "$1" in
  --ckpt) CK="$2"; shift 2;;
  --port) P="$2"; shift 2;;
  --mock) FORCE_MOCK=1; shift;;
  *) shift;;
esac; done

if [ "$FORCE_MOCK" = "1" ] || [ -z "$CK" ] || [ ! -x "$VENV_PY" ]; then
  _log "STAGE 4 serve: MOCK backend on port $P (no checkpoint or --mock)"
  PYBIN=python3 bash "$PROJ/scripts/start_mock_server.sh" "$P"
else
  _log "STAGE 4 serve: pi0.5 checkpoint=$CK on port $P"
  bash "$PROJ/scripts/serve_pi05_checkpoint.sh" --config "$CONFIG_NAME" --ckpt "$CK" --port "$P"
fi
