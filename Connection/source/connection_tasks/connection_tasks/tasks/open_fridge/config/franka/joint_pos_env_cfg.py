# Copyright (c) 2026. Connection project.
# SPDX-License-Identifier: BSD-3-Clause

"""Franka open-fridge env with joint-position control (local assets)."""

from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from connection_tasks.robots.franka import CONNECTION_FRANKA_PANDA_CFG
from connection_tasks.robots.fridge import FRANKA_FRIDGE_READY_QPOS
from connection_tasks.tasks.open_fridge import mdp
from connection_tasks.tasks.open_fridge.open_fridge_env_cfg import (
    FRAME_MARKER_SMALL_CFG,
    OpenFridgeEnvCfg,
)


@configclass
class FrankaOpenFridgeEnvCfg(OpenFridgeEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Set Franka as robot, starting in a gripper-forward "ready" pose in front
        # of the handle so the approach never sweeps the arm through the door.
        self.scene.robot = CONNECTION_FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = dict(FRANKA_FRIDGE_READY_QPOS)

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=1.0,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        # End-effector frame transformer.
        # IMPORTANT: first target frame is the TCP, the others are the fingers.
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="ee_tcp",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.1034)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
                ),
            ],
        )


@configclass
class FrankaOpenFridgeEnvCfg_PLAY(FrankaOpenFridgeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
