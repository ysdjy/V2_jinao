#!/usr/bin/env bash
# Inspect the newest (or a given) HDF5 and print its fields. Run between stage 1 and 2.
#   bash WORKFLOW/inspect.sh                  # newest hdf5
#   bash WORKFLOW/inspect.sh data/raw_hdf5/foo.hdf5
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
require_venv
DEMO="${1:-$(latest_hdf5)}"
[ -z "$DEMO" ] && { echo "No HDF5 found in data/raw_hdf5/. Run WORKFLOW/1_collect.sh first."; exit 1; }
TS="$(date +%Y%m%d_%H%M%S)"
_log "inspecting $DEMO"
"$VENV_PY" "$PROJ/adapters/data_conversion/inspect_hdf5.py" \
  --input "$DEMO" --json-out "$PROJ/logs/inspect_${TS}.json" \
  | tee "$PROJ/logs/inspect_${TS}.txt"
_log "saved logs/inspect_${TS}.{txt,json}"
