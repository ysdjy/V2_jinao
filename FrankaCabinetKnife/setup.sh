#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[setup] checking local assets..."
python - <<PY
from pathlib import Path

project = Path("${PROJECT_DIR}")
required = [
    project / "assets/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd",
    project / "assets/Isaac/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
    project / "assets/Props/Knife_101054/knife.usd",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit("Missing assets:\\n  " + "\\n  ".join(missing))
for path in required:
    print(f"  OK {path}")
PY

echo "[setup] installing franka_cabinet_knife_tasks (editable)..."
python -m pip install -e "${PROJECT_DIR}/source/franka_cabinet_knife_tasks"

echo ""
echo "[setup] done."
echo "  Verify: bash FrankaCabinetKnife/scripts/smoke_test.sh"
echo "  Run:    ./isaaclab.sh -p FrankaCabinetKnife/scripts/state_machine/open_cabinet_knife_sm.py --num_envs 1"

