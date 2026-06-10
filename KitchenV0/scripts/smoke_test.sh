#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" == "base" ]]; then
  echo "Activate your Isaac Lab conda environment first, for example:"
  echo "  conda activate env_isaaclab"
  exit 1
fi

python - <<'PY'
from kitchen_tasks.assets_paths import verify_assets
verify_assets()
print("[smoke] assets ok")
PY

python - <<'PY'
import gymnasium as gym
import kitchen_tasks  # noqa: F401

for task_id in [
    "Kitchen-Fridge-Franka-v0",
    "Kitchen-Fridge-Franka-Play-v0",
    "Kitchen-Fridge-Franka-IK-Abs-v0",
    "Kitchen-Fridge-Franka-IK-Abs-Play-v0",
    "Kitchen-V0-Franka-v0",
    "Kitchen-V0-Franka-Play-v0",
    "Kitchen-V0-Franka-IK-Abs-v0",
    "Kitchen-V0-Franka-IK-Abs-Play-v0",
]:
    assert gym.spec(task_id) is not None
    print(f"[smoke] registered {task_id}")
PY

echo "[smoke] done"
