#!/usr/bin/env bash
# pi0.5 dry-run. Uses the OpenPI venv if present, else falls back to system python+mock.
set -u
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ"
LOG="logs/pi05_dryrun_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

VENV_PY="$PROJ/.venv_openpi/bin/python"
if [ -x "$VENV_PY" ]; then
  echo "[dryrun] trying real OpenPI policy via $VENV_PY" | tee "$LOG"
  # If a checkpoint env is provided via args, forward them; default tries aloha_sim default ckpt.
  "$VENV_PY" scripts/test_pi05_dryrun.py --backend openpi "${@:---env-default}" "${@:+}" 2>&1 | tee -a "$LOG"
  code=${PIPESTATUS[0]}
  if [ "$code" -eq 0 ]; then exit 0; fi
  echo "[dryrun] OpenPI path failed, falling back to mock" | tee -a "$LOG"
fi
echo "[dryrun] running mock backend" | tee -a "$LOG"
python3 scripts/test_pi05_dryrun.py --backend mock 2>&1 | tee -a "$LOG"
