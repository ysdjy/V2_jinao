#!/usr/bin/env bash
# Wrapper around IsaacLab's official scripts/tools/record_demos.py.
# Saves HDF5 demos into pi05_isaacsim_baseline/data/raw_hdf5/.
#
# Usage:
#   bash scripts/collect_demos.sh \
#       --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
#       --teleop_device keyboard \
#       --num_demos 10 \
#       [--enable_cameras] [--device cuda:0] \
#       [--dataset_file data/raw_hdf5/mydemo.hdf5]
#
# Your own teleop device: pass --teleop_device <name>. This script does NOT
# assume a device; it forwards whatever you pass to record_demos.py.
set -u
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "$PROJ/.." && pwd)"
cd "$ROOT"

TASK="Isaac-Stack-Cube-Franka-IK-Rel-v0"
DEVICE_TELEOP="keyboard"
NUM=10
DATASET=""
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --task) TASK="$2"; shift 2;;
    --teleop_device) DEVICE_TELEOP="$2"; shift 2;;
    --num_demos) NUM="$2"; shift 2;;
    --dataset_file) DATASET="$2"; shift 2;;
    *) EXTRA+=("$1"); shift;;
  esac
done

mkdir -p "$PROJ/data/raw_hdf5"
if [ -z "$DATASET" ]; then
  DATASET="$PROJ/data/raw_hdf5/franka_demo_$(date +%Y%m%d_%H%M%S).hdf5"
fi
# make relative dataset paths land inside the project
case "$DATASET" in /*) ;; *) DATASET="$PROJ/$DATASET";; esac

echo "Task:        $TASK"
echo "Teleop:      $DEVICE_TELEOP"
echo "Num demos:   $NUM"
echo "Dataset out: $DATASET"
echo "Extra args:  ${EXTRA[*]:-(none)}"

LOG="$PROJ/logs/collect_demos_$(date +%Y%m%d_%H%M%S).log"
source /home1/banghai/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate env_isaaclab 2>/dev/null || true
./isaaclab.sh -p scripts/tools/record_demos.py \
  --task "$TASK" \
  --teleop_device "$DEVICE_TELEOP" \
  --num_demos "$NUM" \
  --dataset_file "$DATASET" \
  "${EXTRA[@]}" 2>&1 | tee "$LOG"
