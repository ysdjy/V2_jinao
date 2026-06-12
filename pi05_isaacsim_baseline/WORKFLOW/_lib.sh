#!/usr/bin/env bash
# Shared helpers for the WORKFLOW/*.sh stage scripts. Not run directly.
# Resolves project paths and sources pipeline.env.
set -u

WF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$WF_DIR/.." && pwd)"          # pi05_isaacsim_baseline
ROOT="$(cd "$PROJ/.." && pwd)"            # IsaacLab root
VENV_PY="$PROJ/.venv_openpi/bin/python"

# Load central config
# shellcheck disable=SC1091
source "$WF_DIR/pipeline.env"

_log() { echo "[$(date +%H:%M:%S)] $*"; }

# Newest HDF5 in data/raw_hdf5 (override by exporting DEMO=...)
latest_hdf5() {
  ls -t "$PROJ"/data/raw_hdf5/*.hdf5 2>/dev/null | head -1
}

# Newest valid checkpoint step dir (contains _CHECKPOINT_METADATA). Override with CKPT=...
latest_ckpt() {
  find "$PROJ"/policies/checkpoints -name "_CHECKPOINT_METADATA" -printf '%T@ %h\n' 2>/dev/null \
    | sort -rn | head -1 | cut -d' ' -f2-
}

require_venv() {
  if [ ! -x "$VENV_PY" ]; then
    echo "ERROR: OpenPI venv missing ($VENV_PY). Run: bash $PROJ/scripts/setup_openpi_env.sh" >&2
    exit 2
  fi
}

# Activate env_isaaclab for any isaaclab.sh invocation
activate_isaaclab() {
  source /home1/banghai/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate env_isaaclab 2>/dev/null || true
}
