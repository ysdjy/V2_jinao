# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration for the Franka cabinet + knife task."""

import gymnasium as gym

TASK_ID = "FrankaCabinetKnife-Open-Drawer-Franka-IK-Abs-v0"
PLAY_TASK_ID = "FrankaCabinetKnife-Open-Drawer-Franka-IK-Abs-Play-v0"

gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:FrankaCabinetKnifeEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id=PLAY_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:FrankaCabinetKnifeEnvCfg_PLAY",
    },
    disable_env_checker=True,
)

