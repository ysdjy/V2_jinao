# Copyright (c) 2026. KitchenV0 project.
# SPDX-License-Identifier: BSD-3-Clause

"""Franka + fridge only environment.

This is the calibration scene for opening the refrigerator door. Other kitchen
assets are intentionally not spawned so that reachability, collision and the
state machine can be tuned without unrelated geometry.
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass

from kitchen_tasks.robots.fridge import (
    FRIDGE_BODY_LINK,
    FRIDGE_DOOR_JOINT,
    FRIDGE_DOOR_LINK,
    FRIDGE_HANDLE_OFFSET,
    KITCHEN_FRIDGE_CFG,
)

from . import mdp

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


@configclass
class FridgeSceneCfg(InteractiveSceneCfg):
    """Scene with only a Franka and a refrigerator."""

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    fridge: ArticulationCfg = KITCHEN_FRIDGE_CFG.replace(prim_path="{ENV_REGEX_NS}/Fridge")

    fridge_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Fridge/" + FRIDGE_BODY_LINK,
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/KitchenFridgeOnlyFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Fridge/" + FRIDGE_DOOR_LINK,
                name="door_handle",
                offset=OffsetCfg(pos=FRIDGE_HANDLE_OFFSET, rot=(0.5, 0.5, -0.5, -0.5)),
            ),
        ],
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    """Action specifications set by the Franka-specific config."""

    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Minimal observations for state-machine diagnostics."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        fridge_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("fridge", joint_names=[FRIDGE_DOOR_JOINT])},
        )
        rel_ee_handle_distance = ObsTerm(func=mdp.rel_ee_handle_distance)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """No reward terms in this state-machine scene."""


@configclass
class EventCfg:
    """Reset events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class TerminationsCfg:
    """Episode termination terms."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class FridgeSceneEnvCfg(ManagerBasedRLEnvCfg):
    """Franka + fridge only environment config."""

    scene: FridgeSceneCfg = FridgeSceneCfg(num_envs=128, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 1
        self.episode_length_s = 16.0
        self.viewer.eye = (0.35, -2.45, 1.35)
        self.viewer.lookat = (0.52, -0.08, 0.52)
        self.sim.dt = 1 / 60
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
