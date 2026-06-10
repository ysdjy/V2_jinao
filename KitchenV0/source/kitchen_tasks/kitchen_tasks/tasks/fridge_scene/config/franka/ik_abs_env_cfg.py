# Copyright (c) 2026. KitchenV0 project.
# SPDX-License-Identifier: BSD-3-Clause

"""Franka + fridge scene with absolute-pose IK control."""

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from kitchen_tasks.robots.franka import KITCHEN_FRANKA_PANDA_HIGH_PD_CFG
from kitchen_tasks.robots.fridge import FRANKA_FRIDGE_READY_QPOS

from . import joint_pos_env_cfg


@configclass
class FrankaFridgeSceneEnvCfg(joint_pos_env_cfg.FrankaFridgeSceneEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = KITCHEN_FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = dict(FRANKA_FRIDGE_READY_QPOS)
        self.scene.ee_frame.debug_vis = True

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )


@configclass
class FrankaFridgeSceneEnvCfg_PLAY(FrankaFridgeSceneEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
