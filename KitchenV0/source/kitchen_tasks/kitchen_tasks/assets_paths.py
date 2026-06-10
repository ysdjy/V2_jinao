# Copyright (c) 2026. KitchenV0 project.
# SPDX-License-Identifier: BSD-3-Clause

"""Resolve local, clone-relative assets for KitchenV0."""

import os
from pathlib import Path

KITCHEN_ROOT = Path(__file__).resolve().parents[3]
ISAACLAB_ROOT = KITCHEN_ROOT.parent

_assets_dir_env = os.environ.get("KITCHEN_V0_ASSETS_DIR")
ASSETS_DIR = Path(_assets_dir_env) if _assets_dir_env else KITCHEN_ROOT / "assets"

_legacy_assets_dir = ISAACLAB_ROOT / "Connection" / "assets"
_legacy_usd_dir = ISAACLAB_ROOT / "Connection" / "USD"

FRANKA_PANDA_USD = str(
    ASSETS_DIR / "Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd"
    if (ASSETS_DIR / "Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd").is_file()
    else _legacy_assets_dir / "Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd"
)
SEKTION_CABINET_USD = str(
    ASSETS_DIR / "Isaac/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd"
    if (ASSETS_DIR / "Isaac/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd").is_file()
    else _legacy_assets_dir / "Isaac/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd"
)
FRIDGE_USD = str(
    ASSETS_DIR / "Props/Fridge_12252/fridge.usd"
    if (ASSETS_DIR / "Props/Fridge_12252/fridge.usd").is_file()
    else _legacy_assets_dir / "Props/Fridge_12252/fridge.usd"
)
MICROWAVE_URDF = str(
    ASSETS_DIR / "PartNet/7320/mobility_sanitized.urdf"
    if (ASSETS_DIR / "PartNet/7320/mobility_sanitized.urdf").is_file()
    else _legacy_usd_dir / "7320/mobility.urdf"
)
KNIFE_URDF = str(
    ASSETS_DIR / "PartNet/101054/mobility_sanitized.urdf"
    if (ASSETS_DIR / "PartNet/101054/mobility_sanitized.urdf").is_file()
    else _legacy_usd_dir / "101054/mobility.urdf"
)


def verify_assets() -> None:
    """Raise a clear error if required local assets are missing."""

    required = {
        "Franka Panda USD": FRANKA_PANDA_USD,
        "Sektion cabinet USD": SEKTION_CABINET_USD,
        "Fridge USD": FRIDGE_USD,
        "Microwave URDF": MICROWAVE_URDF,
        "Knife URDF": KNIFE_URDF,
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "KitchenV0 is missing required local assets:\n  "
            + "\n  ".join(missing)
            + "\n\nSet KITCHEN_V0_ASSETS_DIR or keep the repository's Connection/assets and Connection/USD folders."
        )
