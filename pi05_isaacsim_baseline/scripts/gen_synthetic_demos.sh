#!/usr/bin/env bash
# Wrapper: generate SYNTHETIC (throwaway) demos for plumbing validation — no human, no GUI.
# Output HDF5 lands in pi05_isaacsim_baseline/data/raw_hdf5/ with the SAME structure as
# record_demos.py output, so the HDF5->LeRobot->train smoke test can be exercised autonomously.
#
# These are NOT task-successful demos and NOT for real training. Real demos come from a human
# at the keyboard (collect_demos.sh) or the GELLO arm later.
#
# Usage:
#   bash scripts/gen_synthetic_demos.sh [--num_demos 3] [--episode_len 120] \
#       [--action_mode scripted|random] [--dataset_file data/raw_hdf5/NAME.hdf5] [--headless]
set -u
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "$PROJ/.." && pwd)"
cd "$ROOT"

NUM=3
EPLEN=120
MODE="scripted"
DATASET=""
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --num_demos) NUM="$2"; shift 2;;
    --episode_len) EPLEN="$2"; shift 2;;
    --action_mode) MODE="$2"; shift 2;;
    --dataset_file) DATASET="$2"; shift 2;;
    *) EXTRA+=("$1"); shift;;
  esac
done

mkdir -p "$PROJ/data/raw_hdf5" "$PROJ/logs"
if [ -z "$DATASET" ]; then
  DATASET="$PROJ/data/raw_hdf5/synthetic_plumbing_$(date +%Y%m%d_%H%M%S).hdf5"
fi
case "$DATASET" in /*) ;; *) DATASET="$PROJ/$DATASET";; esac
# record_demos/IsaacLab refuse to overwrite an existing dataset file; remove a stale one.
[ -f "$DATASET" ] && { echo "[gen] removing existing $DATASET"; rm -f "$DATASET"; }

echo "Task:        Isaac-Stack-Cube-Franka-IK-Rel-v0"
echo "Num demos:   $NUM   episode_len: $EPLEN   mode: $MODE"
echo "Dataset out: $DATASET   (SYNTHETIC / throwaway)"
echo "Extra args:  ${EXTRA[*]:-(none)}"

LOG="$PROJ/logs/gen_synthetic_$(date +%Y%m%d_%H%M%S).log"
source /home1/banghai/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate env_isaaclab 2>/dev/null || true
./isaaclab.sh -p pi05_isaacsim_baseline/scripts/isaaclab/generate_synthetic_demos.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
  --num_demos "$NUM" \
  --episode_len "$EPLEN" \
  --action_mode "$MODE" \
  --dataset_file "$DATASET" \
  "${EXTRA[@]}" 2>&1 | tee "$LOG"
