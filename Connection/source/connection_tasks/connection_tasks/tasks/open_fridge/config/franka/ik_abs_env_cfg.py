# Copyright (c) 2026. Connection project.
# SPDX-License-Identifier: BSD-3-Clause

"""Franka open-fridge env with absolute-pose IK control (used by the state machine)."""

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from connection_tasks.robots.franka import CONNECTION_FRANKA_PANDA_HIGH_PD_CFG
from connection_tasks.robots.fridge import FRANKA_FRIDGE_READY_QPOS

from . import joint_pos_env_cfg


@configclass
class FrankaOpenFridgeEnvCfg(joint_pos_env_cfg.FrankaOpenFridgeEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # stiffer PD controller for better IK tracking; keep the gripper-forward
        # ready pose so the approach never sweeps the arm through the door.
        self.scene.robot = CONNECTION_FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = dict(FRANKA_FRIDGE_READY_QPOS)

        # absolute-pose differential IK action
        # show EE + handle frames when running with GUI for debugging approach paths
        self.scene.ee_frame.debug_vis = True

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )


@configclass
class FrankaOpenFridgeEnvCfg_PLAY(FrankaOpenFridgeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
