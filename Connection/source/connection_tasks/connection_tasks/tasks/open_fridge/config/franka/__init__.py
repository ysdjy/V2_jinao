# Copyright (c) 2026. Connection project.
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration for the Franka open-fridge task (V0)."""

import gymnasium as gym

##
# Joint Position Control
##

gym.register(
    id="Connection-Open-Fridge-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaOpenFridgeEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Connection-Open-Fridge-Franka-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaOpenFridgeEnvCfg_PLAY",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Absolute Pose Control (used by the state machine)
##

gym.register(
    id="Connection-Open-Fridge-Franka-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:FrankaOpenFridgeEnvCfg",
    },
    disable_env_checker=True,
)
