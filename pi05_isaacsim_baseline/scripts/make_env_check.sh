#!/usr/bin/env bash
# Generate the environment detection report -> logs/env_check.txt
set -u
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$PROJ/logs/env_check.txt"
mkdir -p "$PROJ/logs"
export PATH="$HOME/.local/bin:$PATH"

{
  echo "==== ENV CHECK $(date) ===="
  echo "host: $(hostname)  user: $(whoami)  uid: $(id -u)"
  echo
  echo "---- OS ----"; uname -a
  echo
  echo "---- conda ----"; command -v conda && conda env list 2>/dev/null
  echo
  echo "---- IsaacLab ----"
  for c in "/home1/banghai/Documents/IsaacLab" "$HOME/Documents/IsaacLab" "$HOME/IsaacLab"; do
    [ -f "$c/isaaclab.sh" ] && echo "IsaacLab root: $c (isaaclab.sh present)" && break
  done
  echo "env_isaaclab python: $(/home1/banghai/miniconda3/envs/env_isaaclab/bin/python --version 2>&1)"
  echo
  echo "---- GPU / CUDA ----"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null
  echo "nvcc: $(command -v nvcc) $(nvcc --version 2>/dev/null | tail -1)"
  echo
  echo "---- tools ----"
  for t in git uv ffmpeg cmake gcc make curl; do
    printf "  %-8s %s\n" "$t" "$(command -v $t 2>/dev/null || echo MISSING)"
  done
  echo
  echo "---- OpenPI venv ----"
  VENV_PY="$PROJ/.venv_openpi/bin/python"
  if [ -x "$VENV_PY" ]; then echo "venv: $($VENV_PY --version 2>&1)"; else echo "venv: NOT installed yet"; fi
  echo
  echo "---- disk / mem ----"
  df -h /home1 2>/dev/null | tail -1
  free -h 2>/dev/null | head -2
} 2>&1 | tee "$OUT"
echo "Saved -> $OUT"
