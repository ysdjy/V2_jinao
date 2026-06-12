#!/usr/bin/env bash
# Start a policy server (mock or real checkpoint) and run an IsaacLab evaluation.
# Produces logs/eval_policy_*.json and logs/eval_summary_*.json.
#
# Usage:
#   bash scripts/eval_pi05_in_isaaclab.sh \
#       --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
#       --num_rollouts 10 --max_steps 200 \
#       [--config pi05_isaaclab_franka --ckpt <dir>]  # real model
#       [--headless] [--image_mode none]
set -u
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "$PROJ/.." && pwd)"

TASK="Isaac-Stack-Cube-Franka-IK-Rel-v0"
NUM=10
MAXS=200
PORT=8008
IMG=none
CONFIG=""
CKPT=""
HEADLESS="--headless"
while [ $# -gt 0 ]; do
  case "$1" in
    --task) TASK="$2"; shift 2;;
    --num_rollouts) NUM="$2"; shift 2;;
    --max_steps) MAXS="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --image_mode) IMG="$2"; shift 2;;
    --config) CONFIG="$2"; shift 2;;
    --ckpt) CKPT="$2"; shift 2;;
    --no-headless) HEADLESS=""; shift;;
    *) shift;;
  esac
done

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PROJ/logs"

# 1. start server
if [ -n "$CONFIG" ] && [ -n "$CKPT" ]; then
  bash "$PROJ/scripts/serve_pi05_checkpoint.sh" --config "$CONFIG" --ckpt "$CKPT" --port "$PORT"
else
  PYBIN=python3 bash "$PROJ/scripts/start_mock_server.sh" "$PORT"
fi

# 2. run rollout via IsaacLab (activate env_isaaclab so isaaclab.sh uses the right python)
cd "$ROOT"
LOG="$PROJ/logs/eval_run_${TS}.log"
source /home1/banghai/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate env_isaaclab 2>/dev/null || true
./isaaclab.sh -p "$PROJ/scripts/isaaclab/run_policy_in_isaaclab.py" \
    --task "$TASK" $HEADLESS \
    --num_rollouts "$NUM" --max_steps "$MAXS" \
    --policy_port "$PORT" --image_mode "$IMG" 2>&1 | tee "$LOG"
CODE=${PIPESTATUS[0]}

# 3. stop server
bash "$PROJ/scripts/stop_server.sh"
echo "eval exit $CODE  log $LOG"
exit $CODE
