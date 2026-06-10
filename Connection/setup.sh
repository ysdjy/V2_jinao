#!/usr/bin/env bash
# One-command setup for the Connection project on a fresh Isaac Lab install.
#
# Usage (from inside an Isaac Lab clone, with Isaac Sim 5.1 + isaaclab installed):
#   conda activate env_isaaclab
#   bash Connection/setup.sh
#
# What it does:
#   1. (if needed) downloads the offline USD assets into Connection/assets
#   2. installs the connection_tasks extension in editable mode

set -euo pipefail

CONNECTION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. assets
FRANKA_USD="${CONNECTION_DIR}/assets/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd"
if [[ ! -f "${FRANKA_USD}" ]]; then
  echo "[setup] assets missing, downloading..."
  bash "${CONNECTION_DIR}/scripts/download_assets.sh"
else
  echo "[setup] assets already present."
fi

# 2. install extension (editable)
echo "[setup] installing connection_tasks (editable)..."
python -m pip install -e "${CONNECTION_DIR}/source/connection_tasks"

echo ""
echo "[setup] done."
echo "  Verify:    bash Connection/scripts/smoke_test.sh"
echo "  Run (GUI): ./isaaclab.sh -p Connection/scripts/state_machine/open_drawer_sm.py --num_envs 8"
echo "  Run (headless): ./isaaclab.sh -p Connection/scripts/state_machine/open_drawer_sm.py --num_envs 8 --headless"
