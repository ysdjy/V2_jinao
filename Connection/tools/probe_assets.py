# Copyright (c) 2026.
# Probe Nucleus asset URLs and full dependency chains for offline localization.

"""Launch Isaac Sim headless, print the Nucleus root URL and the complete
dependency list (layers + referenced assets) for the Franka and Sektion Cabinet
USD files. This tells us exactly which files must be copied for an offline,
self-contained project.
"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from pxr import Usd, UsdUtils  # noqa: E402

from isaaclab.utils.assets import (  # noqa: E402
    ISAAC_NUCLEUS_DIR,
    ISAACLAB_NUCLEUS_DIR,
    NUCLEUS_ASSET_ROOT_DIR,
    NVIDIA_NUCLEUS_DIR,
)

out_path = "/tmp/asset_deps.txt"
lines = []
lines.append(f"ROOT :: {NUCLEUS_ASSET_ROOT_DIR}")
lines.append(f"NVIDIA :: {NVIDIA_NUCLEUS_DIR}")
lines.append(f"ISAAC :: {ISAAC_NUCLEUS_DIR}")
lines.append(f"ISAACLAB :: {ISAACLAB_NUCLEUS_DIR}")

targets = {
    "franka": f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
    "cabinet": f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
}

for name, path in targets.items():
    lines.append(f"TARGET {name} :: {path}")
    try:
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(path)
        for layer in layers:
            lines.append(f"{name}_LAYER :: {layer.identifier}")
        for asset in assets:
            lines.append(f"{name}_ASSET :: {asset}")
        for u in unresolved:
            lines.append(f"{name}_UNRESOLVED :: {u}")
    except Exception as ex:  # noqa: BLE001
        lines.append(f"{name}_ERROR :: {type(ex).__name__}: {ex}")

with open(out_path, "w") as f:
    f.write("\n".join(lines) + "\n")

simulation_app.close()
