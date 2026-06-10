# Copyright (c) 2026. Connection project.
# SPDX-License-Identifier: BSD-3-Clause

"""Franka multi-skill env with absolute-pose IK control (used by the state machines)."""

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from connection_tasks.robots.franka import CONNECTION_FRANKA_PANDA_HIGH_PD_CFG
from connection_tasks.tasks.multi_skill.multi_skill_env_cfg import (
    CABINET_SCENE_ORIGIN,
)

from . import joint_pos_env_cfg


def _make_high_pd_franka_cfg(prim_path: str, origin: tuple[float, float, float]):
    cfg = CONNECTION_FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path=prim_path)
    cfg.init_state.pos = origin
    return cfg


def _make_ik_action(asset_name: str) -> DifferentialInverseKinematicsActionCfg:
    return DifferentialInverseKinematicsActionCfg(
        asset_name=asset_name,
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
    )


@configclass
class FrankaMultiSkillEnvCfg(joint_pos_env_cfg.FrankaMultiSkillEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.cabinet_robot = _make_high_pd_franka_cfg("{ENV_REGEX_NS}/CabinetRobot", CABINET_SCENE_ORIGIN)

        self.actions.cabinet_arm_action = _make_ik_action("cabinet_robot")


@configclass
class FrankaMultiSkillEnvCfg_PLAY(FrankaMultiSkillEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 8.0
        self.observations.policy.enable_corruption = False
