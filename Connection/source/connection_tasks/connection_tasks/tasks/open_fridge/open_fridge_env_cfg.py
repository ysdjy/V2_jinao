# Copyright (c) 2026. Connection project.
# SPDX-License-Identifier: BSD-3-Clause

"""Open-fridge environment configuration (V0).

Scene: a Franka arm in front of a refrigerator (PartNet 12252) on a ground plane.
The fridge is a fixed-base articulation whose door (``link_0``) rotates about the
revolute hinge ``joint_0``. All heavy assets are loaded from the project's local
``assets`` directory, so the task is fully self-contained / reproducible offline.

The robot and end-effector frame are populated by the robot-specific config
(see ``config/franka``).
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

from connection_tasks.robots.fridge import (
    CONNECTION_FRIDGE_CFG,
    FRIDGE_BODY_LINK,
    FRIDGE_DOOR_JOINT,
    FRIDGE_DOOR_LINK,
    FRIDGE_HANDLE_OFFSET,
)

from . import mdp

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


##
# Scene definition
##


@configclass
class OpenFridgeSceneCfg(InteractiveSceneCfg):
    """Scene with a robot and a refrigerator (robot + ee_frame set by robot config)."""

    # robot + end-effector frame: populated by the robot-specific env cfg
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    # refrigerator articulation (door = joint_0)
    fridge: ArticulationCfg = CONNECTION_FRIDGE_CFG.replace(prim_path="{ENV_REGEX_NS}/Fridge")

    # frame attached to the door handle (state-machine grasp target).
    # source = fridge body; target = the door link with a fixed offset to the handle.
    fridge_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Fridge/" + FRIDGE_BODY_LINK,
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/FridgeFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Fridge/" + FRIDGE_DOOR_LINK,
                name="door_handle",
                offset=OffsetCfg(
                    pos=FRIDGE_HANDLE_OFFSET,
                    rot=(0.5, 0.5, -0.5, -0.5),  # align with end-effector frame (tuned)
                ),
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


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP (set by the robot-specific config)."""

    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        fridge_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("fridge", joint_names=[FRIDGE_DOOR_JOINT])},
        )
        rel_ee_handle_distance = ObsTerm(func=mdp.rel_ee_handle_distance)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

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
class RewardsCfg:
    """Reward terms (empty: the state machine drives the task; kept for RL extension)."""


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class OpenFridgeEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the open-fridge environment."""

    scene: OpenFridgeSceneCfg = OpenFridgeSceneCfg(num_envs=4096, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 1
        self.episode_length_s = 12.0
        self.viewer.eye = (0.3, -2.6, 1.4)
        self.viewer.lookat = (0.5, -0.1, 0.5)
        self.sim.dt = 1 / 60  # 60Hz
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
