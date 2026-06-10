#!/usr/bin/env bash
# Quick checks that do NOT require launching the full simulation:
#   1. local USD assets exist
#   2. connection_tasks imports and the V0/multi-skill tasks are registered in gym
#
# Run from the Isaac Lab root:  bash Connection/scripts/smoke_test.sh

set -euo pipefail

CONNECTION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_DIR="$(cd "${CONNECTION_DIR}/.." && pwd)"

echo "[smoke] checking local assets..."
PYTHONPATH="${CONNECTION_DIR}/source/connection_tasks:${PYTHONPATH:-}" python - <<'PY'
from connection_tasks.assets_paths import (
    verify_assets,
    FRANKA_PANDA_USD,
    SEKTION_CABINET_USD,
    FRIDGE_USD,
    CABINET_44853_USD,
    MICROWAVE_7320_USD,
    KNIFE_101054_USD,
)
verify_assets()
print("  OK franka          :", FRANKA_PANDA_USD)
print("  OK sektion cabinet :", SEKTION_CABINET_USD)
print("  OK fridge          :", FRIDGE_USD)
print("  OK cabinet 44853   :", CABINET_44853_USD)
print("  OK microwave       :", MICROWAVE_7320_USD)
print("  OK knife           :", KNIFE_101054_USD)
PY

echo "[smoke] checking gym registration (launches Isaac Sim, may take ~1 min)..."
cd "${ISAACLAB_DIR}"
TERM=xterm-256color ./isaaclab.sh -p - <<'PY'
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app

import gymnasium as gym
import connection_tasks  # noqa: F401

want = [
    "Connection-Open-Drawer-Franka-v0",
    "Connection-Open-Drawer-Franka-IK-Abs-v0",
    "Connection-Open-Fridge-Franka-v0",
    "Connection-Open-Fridge-Franka-IK-Abs-v0",
    "Connection-Multi-Skill-Franka-v0",
    "Connection-Multi-Skill-Franka-IK-Abs-v0",
    "Connection-Multi-Skill-Franka-IK-Abs-Play-v0",
]
have = set(gym.registry.keys())
missing = []
for t in want:
    present = t in have
    print(("  OK " if present else "  MISSING ") + t, flush=True)
    if not present:
        missing.append(t)

if missing:
    raise SystemExit("Missing gym registrations: " + ", ".join(missing))

app.close()
PY

echo "[smoke] done."
