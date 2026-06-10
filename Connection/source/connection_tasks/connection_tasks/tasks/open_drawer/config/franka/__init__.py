# Copyright (c) 2026. Connection project.
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration for the Franka open-drawer task (V0)."""

import gymnasium as gym

from . import agents

##
# Joint Position Control
##

gym.register(
    id="Connection-Open-Drawer-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaOpenDrawerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenDrawerPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Connection-Open-Drawer-Franka-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaOpenDrawerEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenDrawerPPORunnerCfg",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Absolute Pose Control (used by the state machine)
##

gym.register(
    id="Connection-Open-Drawer-Franka-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:FrankaOpenDrawerEnvCfg",
    },
    disable_env_checker=True,
)
