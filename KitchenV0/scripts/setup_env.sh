#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KITCHEN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ISAACLAB_ROOT="$(cd "${KITCHEN_ROOT}/.." && pwd)"

cd "${ISAACLAB_ROOT}"

if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" == "base" ]]; then
  echo "Please activate your Isaac Lab conda environment first, for example:"
  echo "  conda activate env_isaaclab"
  exit 1
fi

python -m pip install -e "${KITCHEN_ROOT}/source/kitchen_tasks"

echo "KitchenV0 environment is installed."
echo "Run:"
echo "  ./isaaclab.sh -p KitchenV0/scripts/zero_agent.py --task Kitchen-V0-Franka-IK-Abs-Play-v0 --num_envs 1"
