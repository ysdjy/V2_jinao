#!/usr/bin/env bash
# Wrapper around IsaacLab's official scripts/tools/replay_demos.py.
# Usage:
#   bash scripts/replay_demos.sh --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
#       --dataset_file data/raw_hdf5/franka_demo_XXX.hdf5 [extra args...]
set -u
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "$PROJ/.." && pwd)"
cd "$ROOT"

TASK="Isaac-Stack-Cube-Franka-IK-Rel-v0"
DATASET=""
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --task) TASK="$2"; shift 2;;
    --dataset_file) DATASET="$2"; shift 2;;
    *) EXTRA+=("$1"); shift;;
  esac
done
if [ -z "$DATASET" ]; then echo "ERROR: --dataset_file required"; exit 1; fi
case "$DATASET" in /*) ;; *) DATASET="$PROJ/$DATASET";; esac

LOG="$PROJ/logs/replay_demos_$(date +%Y%m%d_%H%M%S).log"
source /home1/banghai/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate env_isaaclab 2>/dev/null || true
./isaaclab.sh -p scripts/tools/replay_demos.py \
  --task "$TASK" --dataset_file "$DATASET" "${EXTRA[@]}" 2>&1 | tee "$LOG"
