# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Franka configs backed by the project's local USD copy."""

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG, FRANKA_PANDA_HIGH_PD_CFG

from franka_cabinet_knife_tasks.assets_paths import FRANKA_PANDA_USD

FRANKA_CABINET_KNIFE_PANDA_CFG = FRANKA_PANDA_CFG.copy()
FRANKA_CABINET_KNIFE_PANDA_CFG.spawn.usd_path = FRANKA_PANDA_USD
"""Official Franka Panda config with the USD path redirected to this project."""

FRANKA_CABINET_KNIFE_PANDA_HIGH_PD_CFG = FRANKA_PANDA_HIGH_PD_CFG.copy()
FRANKA_CABINET_KNIFE_PANDA_HIGH_PD_CFG.spawn.usd_path = FRANKA_PANDA_USD
"""Official high-PD Franka Panda config with the USD path redirected to this project."""

