# Copyright (c) 2026. Connection project.
# SPDX-License-Identifier: BSD-3-Clause

"""Franka multi-skill env with joint-position control (local assets)."""

from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from connection_tasks.robots.franka import CONNECTION_FRANKA_PANDA_CFG
from connection_tasks.tasks.multi_skill import mdp
from connection_tasks.tasks.multi_skill.multi_skill_env_cfg import (
    CABINET_SCENE_ORIGIN,
    FRAME_MARKER_SMALL_CFG,
    MultiSkillEnvCfg,
)


def _make_franka_cfg(prim_path: str, origin: tuple[float, float, float]):
    cfg = CONNECTION_FRANKA_PANDA_CFG.replace(prim_path=prim_path)
    cfg.init_state.pos = origin
    return cfg


def _make_ee_frame(robot_prim: str, marker_prim: str) -> FrameTransformerCfg:
    return FrameTransformerCfg(
        prim_path=f"{robot_prim}/panda_link0",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path=marker_prim),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{robot_prim}/panda_hand",
                name="ee_tcp",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.1034)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{robot_prim}/panda_leftfinger",
                name="tool_leftfinger",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{robot_prim}/panda_rightfinger",
                name="tool_rightfinger",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
            ),
        ],
    )


def _make_arm_action(asset_name: str) -> mdp.JointPositionActionCfg:
    return mdp.JointPositionActionCfg(
        asset_name=asset_name,
        joint_names=["panda_joint.*"],
        scale=1.0,
        use_default_offset=True,
    )


def _make_gripper_action(asset_name: str) -> mdp.BinaryJointPositionActionCfg:
    return mdp.BinaryJointPositionActionCfg(
        asset_name=asset_name,
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class FrankaMultiSkillEnvCfg(MultiSkillEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.cabinet_robot = _make_franka_cfg("{ENV_REGEX_NS}/CabinetRobot", CABINET_SCENE_ORIGIN)

        self.scene.cabinet_ee_frame = _make_ee_frame(
            "{ENV_REGEX_NS}/CabinetRobot", "/Visuals/CabinetEndEffectorFrameTransformer"
        )

        self.actions.cabinet_arm_action = _make_arm_action("cabinet_robot")
        self.actions.cabinet_gripper_action = _make_gripper_action("cabinet_robot")
