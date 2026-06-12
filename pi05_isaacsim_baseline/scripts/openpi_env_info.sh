#!/usr/bin/env bash
# Report on the OpenPI venv. Output -> logs/openpi_env_info.txt
set -u
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$PROJ/logs/openpi_env_info.txt"
VENV_PY="$PROJ/.venv_openpi/bin/python"
mkdir -p "$PROJ/logs"
export PATH="$HOME/.local/bin:$PATH"

{
  echo "==== openpi_env_info $(date) ===="
  echo "uv: $(command -v uv 2>/dev/null) $(uv --version 2>/dev/null)"
  if [ ! -x "$VENV_PY" ]; then
    echo "OpenPI venv NOT found at $VENV_PY"
    echo "Run scripts/setup_openpi_env.sh"
    exit 0
  fi
  echo "python: $($VENV_PY --version 2>&1)"
  "$VENV_PY" - <<'PYEOF'
import importlib
mods = ["openpi","openpi_client","jax","jaxlib","flax","torch","lerobot","numpy","transformers","tyro"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:14s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  {m:14s} MISSING ({type(e).__name__})")
try:
    import torch; print("torch.cuda.is_available:", torch.cuda.is_available(),
                        "device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
except Exception as e:
    print("torch cuda check failed:", e)
try:
    import jax; print("jax devices:", jax.devices())
except Exception as e:
    print("jax devices: n/a", e)
# enumerate available openpi configs / pi0.5 names
try:
    from openpi.training import config as c
    names = [x.name for x in c._CONFIGS]
    pis = [n for n in names if "pi0" in n or "pi05" in n]
    print("openpi pi0/pi0.5 configs:", pis)
except Exception as e:
    print("could not list openpi configs:", e)
PYEOF
} 2>&1 | tee "$OUT"
echo "Saved -> $OUT"
