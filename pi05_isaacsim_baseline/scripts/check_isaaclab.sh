#!/usr/bin/env bash
# Locate IsaacLab, list Franka tasks, find record/replay/teleop scripts.
# Output -> logs/check_isaaclab.txt
set -u
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$PROJ/logs/check_isaaclab.txt"
mkdir -p "$PROJ/logs"

# Locate IsaacLab root
ROOT=""
for c in "/home1/banghai/Documents/IsaacLab" "/home/banghai/Documents/IsaacLab" \
         "/home1/banghai/IsaacLab" "/home/banghai/IsaacLab" "$HOME/Documents/IsaacLab" "$HOME/IsaacLab"; do
  if [ -f "$c/isaaclab.sh" ]; then ROOT="$c"; break; fi
done

{
  echo "==== IsaacLab check $(date) ===="
  if [ -z "$ROOT" ]; then echo "ERROR: IsaacLab root not found"; exit 0; fi
  echo "IsaacLab root: $ROOT"
  echo "isaaclab.sh: $([ -f "$ROOT/isaaclab.sh" ] && echo present || echo MISSING)"
  echo "record_demos.py: $(ls "$ROOT"/scripts/tools/record_demos.py 2>/dev/null || echo MISSING)"
  echo "replay_demos.py: $(ls "$ROOT"/scripts/tools/replay_demos.py 2>/dev/null || echo MISSING)"
  echo "teleop examples:"
  find "$ROOT/scripts" -iname "*teleop*" 2>/dev/null | sed 's/^/  /' | head
  echo
  echo "---- Franka task ids (grep gym.register) ----"
  grep -RhoE "Isaac-[A-Za-z0-9-]*Franka[A-Za-z0-9-]*" "$ROOT/source" 2>/dev/null | sort -u | head -80
  echo
  echo "---- Stack/Lift task ids ----"
  grep -RhoE "Isaac-(Stack|Lift)[A-Za-z0-9-]*" "$ROOT/source" 2>/dev/null | sort -u | head -40
} 2>&1 | tee "$OUT"
echo "Saved -> $OUT"
