# Copyright (c) 2026. KitchenV0 project.
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration for the Franka + fridge only scene."""

import gymnasium as gym

gym.register(
    id="Kitchen-Fridge-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaFridgeSceneEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Kitchen-Fridge-Franka-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaFridgeSceneEnvCfg_PLAY",
    },
    disable_env_checker=True,
)

gym.register(
    id="Kitchen-Fridge-Franka-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:FrankaFridgeSceneEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Kitchen-Fridge-Franka-IK-Abs-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:FrankaFridgeSceneEnvCfg_PLAY",
    },
    disable_env_checker=True,
)
