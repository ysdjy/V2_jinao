#!/usr/bin/env bash
# One-shot smoke test of the whole pipeline (mock-first, fault tolerant).
# This is a thin wrapper over scripts/nightly_autorun.sh kept at project root
# for convenience, plus a fast-path that does NOT require IsaacSim if you pass
# --no-isaac.
#
# Usage:
#   bash run_all_smoke_test.sh            # full (includes IsaacLab rollout)
#   bash run_all_smoke_test.sh --no-isaac # skip the heavy IsaacSim rollout
set -u
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJ"
mkdir -p logs
TS="$(date +%Y%m%d_%H%M%S)"

if [ "${1:-}" = "--no-isaac" ]; then
  LOG="logs/smoke_noisaac_${TS}.log"
  echo "==== smoke (no-isaac) $TS ====" | tee "$LOG"
  bash scripts/make_env_check.sh            >> "$LOG" 2>&1; echo "env_check: $?"            | tee -a "$LOG"
  bash scripts/run_tests.sh python3         >> "$LOG" 2>&1; echo "unit_tests: $?"           | tee -a "$LOG"
  PYBIN=python3 bash scripts/start_mock_server.sh 8008 >> "$LOG" 2>&1; echo "mock_server: $?" | tee -a "$LOG"
  bash scripts/test_pi05_dryrun.sh          >> "$LOG" 2>&1; echo "dryrun: $?"               | tee -a "$LOG"
  bash scripts/stop_server.sh               >> "$LOG" 2>&1
  python3 scripts/update_status.py          >> "$LOG" 2>&1
  echo "smoke (no-isaac) done -> $LOG"
  exit 0
fi

exec bash scripts/nightly_autorun.sh
