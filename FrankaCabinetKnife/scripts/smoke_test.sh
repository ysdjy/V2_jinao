#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
PACKAGE_DIR="${PROJECT_DIR}/source/franka_cabinet_knife_tasks"

echo "[smoke] checking local assets..."
ASSETS_PATHS_FILE="${PACKAGE_DIR}/franka_cabinet_knife_tasks/assets_paths.py" python - <<'PY'
import importlib.util
import os

spec = importlib.util.spec_from_file_location("assets_paths", os.environ["ASSETS_PATHS_FILE"])
assets_paths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets_paths)

assets_paths.verify_assets()
print("  OK franka          :", assets_paths.FRANKA_PANDA_USD)
print("  OK sektion cabinet :", assets_paths.SEKTION_CABINET_USD)
print("  OK knife           :", assets_paths.KNIFE_101054_USD)
PY

echo "[smoke] checking gym registration (launches Isaac Sim, may take ~1 min)..."
cd "${ISAACLAB_DIR}"
PYTHONPATH="${PACKAGE_DIR}:${PYTHONPATH:-}" TERM=xterm-256color ./isaaclab.sh -p - <<'PY'
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import gymnasium as gym
import franka_cabinet_knife_tasks  # noqa: F401

want = [
    "FrankaCabinetKnife-Open-Drawer-Franka-IK-Abs-v0",
    "FrankaCabinetKnife-Open-Drawer-Franka-IK-Abs-Play-v0",
]
have = set(gym.registry.keys())
missing = []
for task_id in want:
    present = task_id in have
    print(("  OK " if present else "  MISSING ") + task_id, flush=True)
    if not present:
        missing.append(task_id)

if missing:
    raise SystemExit("Missing gym registrations: " + ", ".join(missing))

app.close()
PY

echo "[smoke] done."
