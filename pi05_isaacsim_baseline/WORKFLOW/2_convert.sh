#!/usr/bin/env bash
# STAGE 2 — Store/convert: HDF5 -> normalized + LeRobot dataset (training format).
# Uses the newest HDF5 by default; output goes to data/lerobot/$REPO_ID.
#
#   bash WORKFLOW/2_convert.sh                       # newest hdf5 -> data/lerobot/$REPO_ID
#   bash WORKFLOW/2_convert.sh --input data/raw_hdf5/foo.hdf5 --repo-id bar
#   SYNTH_IMAGES=224 bash WORKFLOW/2_convert.sh      # state-only data: inject zero images
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
require_venv

DEMO="$(latest_hdf5)"; RID="$REPO_ID"
while [ $# -gt 0 ]; do case "$1" in
  --input) DEMO="$2"; shift 2;;
  --repo-id) RID="$2"; shift 2;;
  *) shift;;
esac; done
[ -z "$DEMO" ] && { echo "No HDF5 found. Run WORKFLOW/1_collect.sh first."; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
OUT="data/lerobot/${RID}"
SYNTH=(); [ "${SYNTH_IMAGES:-0}" -gt 0 ] 2>/dev/null && SYNTH=(--synthesize-images "$SYNTH_IMAGES")

_log "STAGE 2 convert: $DEMO -> $OUT  (synth_images=${SYNTH_IMAGES:-0})"
"$VENV_PY" "$PROJ/adapters/data_conversion/hdf5_to_lerobot.py" \
  --input "$DEMO" --output "$PROJ/$OUT" \
  --config "$PROJ/configs/dataset_mapping_isaaclab_franka.yaml" \
  --task_instruction "$TASK_INSTRUCTION" \
  --report-out "$PROJ/logs/convert_report_${TS}.json" "${SYNTH[@]}"
echo
"$VENV_PY" "$PROJ/adapters/data_conversion/validate_lerobot_dataset.py" \
  --input "$PROJ/data/processed/normalized_dataset/${RID}" 2>/dev/null || true
_log "done. dataset=$OUT  report=logs/convert_report_${TS}.json  Next: bash WORKFLOW/3_train.sh --repo-id $RID"
