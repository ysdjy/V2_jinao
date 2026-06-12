#!/usr/bin/env bash
# STAGE 1 — Teleoperation data collection.
# Drive the Franka with your teleop device and record SUCCESSFUL demos to HDF5.
# Config comes from pipeline.env; override per-run with flags.
#
#   bash WORKFLOW/1_collect.sh                          # uses pipeline.env defaults
#   bash WORKFLOW/1_collect.sh --device spacemouse --num 20
#   bash WORKFLOW/1_collect.sh --name franka_real_batch1   # custom hdf5 name
#
# A HUMAN drives. Click the 3D viewport first so it has keyboard focus.
# Keyboard controls: W/S X · A/D Y · Q/E Z · Z/X rotX · T/G rotY · C/V rotZ ·
#                    K toggle gripper · R reset/abort episode (discards it).
# Only successful episodes are saved; exits after NUM_DEMOS successes.
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

DEV="$TELEOP_DEVICE"; NUM="$NUM_DEMOS"; NAME=""; CAMS="$ENABLE_CAMERAS"
while [ $# -gt 0 ]; do case "$1" in
  --device) DEV="$2"; shift 2;;
  --num) NUM="$2"; shift 2;;
  --name) NAME="$2"; shift 2;;
  --cameras) CAMS="$2"; shift 2;;
  *) shift;;
esac; done

STAMP="$(date +%Y%m%d_%H%M%S)"
FILE="${NAME:-${REPO_ID}_${STAMP}}"
DATASET="data/raw_hdf5/${FILE}.hdf5"
EXTRA=(); [ "$CAMS" = "1" ] && EXTRA+=(--enable_cameras)

_log "STAGE 1 collect: task=$TASK device=$DEV num=$NUM cameras=$CAMS -> $DATASET"
bash "$PROJ/scripts/collect_demos.sh" \
  --task "$TASK" --teleop_device "$DEV" --num_demos "$NUM" \
  --dataset_file "$DATASET" "${EXTRA[@]}"
_log "done. Next: bash WORKFLOW/inspect.sh   then   bash WORKFLOW/2_convert.sh"
