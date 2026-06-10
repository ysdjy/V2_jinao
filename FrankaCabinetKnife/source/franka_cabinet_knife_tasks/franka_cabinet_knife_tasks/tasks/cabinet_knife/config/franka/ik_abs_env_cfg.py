# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Franka IK-Abs config for the official cabinet scene with a knife added."""

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet.config.franka.ik_abs_env_cfg import (
    FrankaCabinetEnvCfg as OfficialFrankaCabinetEnvCfg,
)

from franka_cabinet_knife_tasks.assets_paths import SEKTION_CABINET_USD
from franka_cabinet_knife_tasks.robots.franka import FRANKA_CABINET_KNIFE_PANDA_HIGH_PD_CFG
from franka_cabinet_knife_tasks.tasks.cabinet_knife.cabinet_knife_env_cfg import CabinetKnifeSceneCfg


@configclass
class FrankaCabinetKnifeEnvCfg(OfficialFrankaCabinetEnvCfg):
    """Official Franka open-drawer IK config plus a knife beside the cabinet."""

    scene: CabinetKnifeSceneCfg = CabinetKnifeSceneCfg(num_envs=4096, env_spacing=2.0)

    def __post_init__(self):
        super().__post_init__()

        # Keep the official robot/cabinet settings, but resolve the heavy USD files locally.
        self.scene.robot = FRANKA_CABINET_KNIFE_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.cabinet.spawn.usd_path = SEKTION_CABINET_USD
        self.episode_length_s = 40.0


@configclass
class FrankaCabinetKnifeEnvCfg_PLAY(FrankaCabinetKnifeEnvCfg):
    """Smaller scene variant for interactive display."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
