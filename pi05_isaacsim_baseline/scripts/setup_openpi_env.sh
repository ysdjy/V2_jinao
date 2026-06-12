#!/usr/bin/env bash
# Create an ISOLATED OpenPI / pi0.5 environment using uv. Does NOT touch
# env_isaaclab or system python. The venv lives at third_party/openpi/.venv and
# is symlinked to .venv_openpi at the project root for convenience.
#
# Safe to re-run. Logs to logs/setup_openpi_env_*.log (when invoked via tee).
set -u
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI="$PROJ/third_party/openpi"
export GIT_LFS_SKIP_SMUDGE=1

echo "==== setup_openpi_env $(date) ===="
echo "Project: $PROJ"
echo "OpenPI:  $OPENPI"

# ---- 1. ensure uv ----
if ! command -v uv >/dev/null 2>&1; then
  if [ -x "$HOME/.local/bin/uv" ]; then
    export PATH="$HOME/.local/bin:$PATH"
  else
    echo "[setup] installing uv to ~/.local/bin ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh || { echo "uv install FAILED"; exit 2; }
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi
echo "[setup] uv: $(command -v uv) $(uv --version 2>/dev/null)"

# ---- 2. ensure openpi sources / submodules ----
if [ ! -f "$OPENPI/pyproject.toml" ]; then
  echo "[setup] cloning openpi ..."
  git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git "$OPENPI" \
    || git clone https://github.com/Physical-Intelligence/openpi.git "$OPENPI"
fi
cd "$OPENPI" || exit 2
# init submodules if the lock needs them (best effort)
git submodule update --init --recursive 2>/dev/null || echo "[setup] (submodule init skipped/failed - continuing)"

# ---- 3. sync deps ----
echo "[setup] uv sync (this downloads JAX/torch and is slow) ..."
if uv sync; then
  echo "[setup] uv sync OK"
else
  echo "[setup] uv sync failed; trying without locked extras ..."
  uv sync --inexact || echo "[setup] uv sync still failing - see log"
fi
echo "[setup] uv pip install -e . ..."
uv pip install -e . || echo "[setup] editable install failed - see log"

# ---- 4. convenience symlink ----
if [ -d "$OPENPI/.venv" ]; then
  ln -sfn "$OPENPI/.venv" "$PROJ/.venv_openpi"
  echo "[setup] symlinked $PROJ/.venv_openpi -> $OPENPI/.venv"
fi

# ---- 5. verify ----
PY="$OPENPI/.venv/bin/python"
if [ -x "$PY" ]; then
  echo "[setup] python: $($PY --version 2>&1)"
  "$PY" - <<'PYEOF'
import importlib, sys
for m in ["openpi", "openpi_client", "jax", "torch", "lerobot", "numpy"]:
    try:
        mod = importlib.import_module(m)
        print(f"  import {m:12s} OK  ({getattr(mod,'__version__','?')})")
    except Exception as e:
        print(f"  import {m:12s} FAIL ({type(e).__name__}: {e})")
try:
    import jax
    print("  jax devices:", jax.devices())
except Exception as e:
    print("  jax devices: n/a", e)
try:
    import torch
    print("  torch cuda:", torch.cuda.is_available())
except Exception:
    pass
PYEOF
else
  echo "[setup] ERROR: venv python not found at $PY"
fi
echo "==== setup_openpi_env done $(date) ===="
