#!/usr/bin/env bash
# Run unit tests that do NOT require IsaacSim. Records results to logs/.
# Usage: bash scripts/run_tests.sh [python_bin]
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
PY="${1:-python3}"
LOG="logs/run_tests_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "Using python: $PY ($($PY --version 2>&1))" | tee "$LOG"
FAIL=0
for t in test_policy_schema test_safety_filter test_mock_policy_server test_hdf5_inspect; do
  echo "========== $t ==========" | tee -a "$LOG"
  ( cd tests && "$PY" "$t.py" ) 2>&1 | tee -a "$LOG"
  code=${PIPESTATUS[0]}
  echo "$t exit code: $code" | tee -a "$LOG"
  [ "$code" -ne 0 ] && FAIL=1
done
echo "==========================" | tee -a "$LOG"
if [ "$FAIL" -eq 0 ]; then echo "ALL NON-ISAAC TESTS PASSED" | tee -a "$LOG"; else echo "SOME TESTS FAILED" | tee -a "$LOG"; fi
exit $FAIL
