# Copyright (c) 2026. Connection project.
# SPDX-License-Identifier: BSD-3-Clause

"""Resolve paths to the project's local (offline) USD assets.

All assets live under ``Connection/assets`` mirroring the Nucleus relative layout,
so the USD files' internal relative references resolve without network access.

The assets directory can be overridden with the ``CONNECTION_ASSETS_DIR`` env var,
which is useful if the assets are relocated on a different machine.
"""

import os
from pathlib import Path

# .../Connection/source/connection_tasks/connection_tasks/assets_paths.py
#   parents[0] = connection_tasks (package)
#   parents[1] = connection_tasks (pip project root)
#   parents[2] = source
#   parents[3] = Connection  (project root)
CONNECTION_ROOT = Path(__file__).resolve().parents[3]

ASSETS_DIR = Path(os.environ.get("CONNECTION_ASSETS_DIR", str(CONNECTION_ROOT / "assets")))

FRANKA_PANDA_USD = str(ASSETS_DIR / "Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd")
SEKTION_CABINET_USD = str(ASSETS_DIR / "Isaac/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd")
# Fridge converted from PartNet-Mobility 12252 (Refrigerator) via tools/convert_urdf.py
FRIDGE_USD = str(ASSETS_DIR / "Props/Fridge_12252/fridge.usd")
# Three-drawer storage cabinet converted from PartNet-Mobility 44853.
CABINET_44853_USD = str(ASSETS_DIR / "Props/Cabinet_44853/cabinet.usd")
# Microwave converted from PartNet-Mobility 7320.
MICROWAVE_7320_USD = str(ASSETS_DIR / "Props/Microwave_7320/microwave.usd")
# Folding knife converted from PartNet-Mobility 101054.
KNIFE_101054_USD = str(ASSETS_DIR / "Props/Knife_101054/knife.usd")


def verify_assets() -> None:
    """Raise a clear error if required USD assets are missing."""
    required_assets = (
        FRANKA_PANDA_USD,
        SEKTION_CABINET_USD,
        FRIDGE_USD,
        CABINET_44853_USD,
        MICROWAVE_7320_USD,
        KNIFE_101054_USD,
    )
    missing = [p for p in required_assets if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            "Missing local USD assets:\n  "
            + "\n  ".join(missing)
            + "\n\nRun 'bash Connection/scripts/download_assets.sh' for the Isaac/Nucleus assets. "
            "For PartNet assets, first run 'python Connection/tools/prepare_partnet_urdf.py "
            "Connection/USD/<asset_id>' and then convert the generated mobility_isaac.urdf with "
            "'./isaaclab.sh -p scripts/tools/convert_urdf.py ...'. "
            "Alternatively set CONNECTION_ASSETS_DIR to point at an existing assets directory."
        )
