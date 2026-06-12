#!/usr/bin/env bash
# Fault-tolerant nightly orchestrator. Each step is best-effort: a failure is
# logged and the run continues. Final summary + STATUS.md update at the end.
#
# Usage: bash scripts/nightly_autorun.sh
set -u
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "$PROJ/.." && pwd)"
cd "$PROJ"
export PATH="$HOME/.local/bin:$PATH"
LOG="$PROJ/logs/nightly_autorun.log"
mkdir -p logs
echo "########## NIGHTLY AUTORUN $(date) ##########" | tee "$LOG"

VENV_PY="$PROJ/.venv_openpi/bin/python"
PORT=8008
declare -A RESULT

run_step() {
  local name="$1"; local cmd="$2"
  echo "========== $name ==========" | tee -a "$LOG"
  bash -lc "$cmd" >> "$LOG" 2>&1
  local code=$?
  echo "$name exit code: $code" | tee -a "$LOG"
  RESULT[$name]=$code
  return 0
}

# 1. env check
run_step "env_check" "bash '$PROJ/scripts/make_env_check.sh'"
# 2. project structure check
run_step "structure_check" "ls -R '$PROJ' | head -200"
# 3. openpi info (install assumed already done; report only)
run_step "openpi_info" "bash '$PROJ/scripts/openpi_env_info.sh'"
# 4. non-isaac unit tests
run_step "unit_tests" "bash '$PROJ/scripts/run_tests.sh' python3"
# 5. mock server up + dry-run
run_step "mock_server" "PYBIN=python3 bash '$PROJ/scripts/start_mock_server.sh' $PORT"
run_step "pi05_dryrun" "bash '$PROJ/scripts/test_pi05_dryrun.sh'"
# 6. isaaclab availability
run_step "check_isaaclab" "bash '$PROJ/scripts/check_isaaclab.sh'"
# 7. isaaclab mock rollout (1 ep) -- heavy; may take minutes to launch sim
run_step "isaac_rollout" "source /home1/banghai/miniconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate env_isaaclab 2>/dev/null; cd '$ROOT' && ./isaaclab.sh -p '$PROJ/scripts/isaaclab/run_policy_in_isaaclab.py' --task Isaac-Stack-Cube-Franka-IK-Rel-v0 --headless --num_rollouts 1 --max_steps 60 --policy_port $PORT --image_mode none"
bash "$PROJ/scripts/stop_server.sh" >> "$LOG" 2>&1

# 8-11. dataset-dependent steps
DEMO="$(ls -t "$PROJ"/data/raw_hdf5/*.hdf5 2>/dev/null | head -1)"
if [ -n "${DEMO:-}" ]; then
  run_step "hdf5_inspect" "${VENV_PY:-python3} '$PROJ/adapters/data_conversion/inspect_hdf5.py' --input '$DEMO' --json-out '$PROJ/logs/hdf5_summary.json'"
  run_step "hdf5_to_lerobot" "${VENV_PY:-python3} '$PROJ/adapters/data_conversion/hdf5_to_lerobot.py' --input '$DEMO' --output '$PROJ/data/lerobot/franka_stack_cube_pi05' --config '$PROJ/configs/dataset_mapping_isaaclab_franka.yaml' --report-out '$PROJ/logs/conversion_report.json'"
  DSN="$PROJ/data/processed/normalized_dataset/franka_stack_cube_pi05"
  run_step "dataset_quality" "python3 '$PROJ/scripts/analyze_dataset.py' --input '$DSN'"
  if [ -x "$VENV_PY" ]; then
    run_step "train_smoke" "bash '$PROJ/scripts/train_pi05.sh' --repo-id franka_stack_cube_pi05 --steps 10 --batch-size 1"
  else
    echo "train_smoke SKIPPED (no openpi venv)" | tee -a "$LOG"; RESULT[train_smoke]="skip"
  fi
else
  echo "No demo HDF5 found -> skipping inspect/convert/quality/train steps" | tee -a "$LOG"
  for s in hdf5_inspect hdf5_to_lerobot dataset_quality train_smoke; do RESULT[$s]="skip(no-demo)"; done
fi

# 12. checkpoint serving smoke (if a checkpoint exists)
CKPT="$(ls -td "$PROJ"/policies/checkpoints/*/ 2>/dev/null | head -1)"
if [ -n "${CKPT:-}" ] && [ -x "$VENV_PY" ]; then
  run_step "serve_ckpt_smoke" "bash '$PROJ/scripts/serve_pi05_checkpoint.sh' --config pi05_isaaclab_franka --ckpt '$CKPT' --port $PORT ; sleep 2; curl -fsS http://127.0.0.1:$PORT/health; bash '$PROJ/scripts/stop_server.sh'"
else
  RESULT[serve_ckpt_smoke]="skip(no-ckpt)"
fi

# 13. summary
echo "########## SUMMARY ##########" | tee -a "$LOG"
for k in "${!RESULT[@]}"; do echo "  $k = ${RESULT[$k]}" | tee -a "$LOG"; done
python3 "$PROJ/scripts/update_status.py" >> "$LOG" 2>&1 || echo "status update skipped" | tee -a "$LOG"
echo "########## NIGHTLY AUTORUN DONE $(date) ##########" | tee -a "$LOG"
