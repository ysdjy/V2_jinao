# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Resolve clone-local USD assets for FrankaCabinetKnife."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ASSETS_DIR = Path(os.environ.get("FRANKA_CABINET_KNIFE_ASSETS_DIR", str(PROJECT_ROOT / "assets")))

FRANKA_PANDA_USD = str(ASSETS_DIR / "Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd")
SEKTION_CABINET_USD = str(ASSETS_DIR / "Isaac/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd")
KNIFE_101054_USD = str(ASSETS_DIR / "Props/Knife_101054/knife.usd")


def verify_assets() -> None:
    """Raise a clear error if a required USD asset is missing."""

    required = {
        "Franka Panda USD": FRANKA_PANDA_USD,
        "Sektion cabinet USD": SEKTION_CABINET_USD,
        "Knife 101054 USD": KNIFE_101054_USD,
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "FrankaCabinetKnife is missing required local assets:\n  "
            + "\n  ".join(missing)
            + "\n\nSet FRANKA_CABINET_KNIFE_ASSETS_DIR or keep the project's assets folder in place."
        )

