# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Selected-drawer-conditioned custom drawer RL task (method B, joint-position action).

Reuses the method-B joint-position scene (custom Cabinet_44853 + Franka, scale=1.0 joint action,
high-PD robot — same robot/action the state machine deploys with) and replaces the stack MDP with a
selected-drawer MDP: each episode samples a target drawer (among the functional drawers) and the
policy must open THAT drawer.

Observation is 31-d in the official Franka open-drawer order so an official checkpoint can be loaded
for fine-tuning. A single policy handles all drawers (the target is expressed implicitly via the
selected handle relative pose + selected drawer joint state).
"""

from __future__ import annotations

from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

from . import custom_drawer_mdp as cd_mdp
from .stack_joint_policy_env_cfg import FrankaCubeStackJointPolicyEnvCfg

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # order matches the official open-drawer policy (31-d) for checkpoint compatibility
        joint_pos = ObsTerm(func=base_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=base_mdp.joint_vel_rel)
        cabinet_joint_pos = ObsTerm(func=cd_mdp.selected_drawer_joint_pos)
        cabinet_joint_vel = ObsTerm(func=cd_mdp.selected_drawer_joint_vel)
        rel_ee_drawer_distance = ObsTerm(func=cd_mdp.selected_rel_ee_drawer_distance)
        actions = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    approach_ee_handle = RewTerm(func=cd_mdp.selected_approach_ee_handle, weight=2.0, params={"threshold": 0.2})
    align_ee_handle = RewTerm(func=cd_mdp.selected_align_ee_handle, weight=0.5)
    approach_gripper_handle = RewTerm(func=cd_mdp.selected_approach_gripper_handle, weight=5.0, params={"offset": 0.04})
    align_grasp_around_handle = RewTerm(func=cd_mdp.selected_align_grasp_around_handle, weight=0.125)
    grasp_handle = RewTerm(
        func=cd_mdp.selected_grasp_handle,
        weight=0.5,
        params={"threshold": 0.03, "open_joint_pos": 0.04, "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_finger_.*"])},
    )
    open_drawer_bonus = RewTerm(func=cd_mdp.selected_open_drawer_bonus, weight=7.5)
    multi_stage_open_drawer = RewTerm(func=cd_mdp.selected_multi_stage_open_drawer, weight=1.0)
    wrong_drawer_penalty = RewTerm(func=cd_mdp.wrong_drawer_open_penalty, weight=-1.0)
    action_rate_l2 = RewTerm(func=base_mdp.action_rate_l2, weight=-1e-2)
    joint_vel = RewTerm(func=base_mdp.joint_vel_l2, weight=-1e-4)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    success = DoneTerm(func=cd_mdp.selected_drawer_opened, params={"threshold": 0.20})


@configclass
class EventCfg:
    init_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={"default_pose": [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400]},
    )
    randomize_franka_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={"mean": 0.0, "std": 0.02, "asset_cfg": SceneEntityCfg("robot")},
    )
    reset_select_drawer = EventTerm(func=cd_mdp.reset_select_drawer, mode="reset")


@configclass
class FrankaCustomDrawerSelectedEnvCfg(FrankaCubeStackJointPolicyEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # declutter: this is a drawer task, remove the stack cubes / knife
        self.scene.cube_1 = None
        self.scene.cube_2 = None
        self.scene.cube_3 = None
        self.scene.knife = None
        self.scene.knife_handle_proxy = None

        # handle frames for the functional drawers (order == custom_drawer_config.FUNCTIONAL_DRAWERS)
        marker = FRAME_MARKER_CFG.copy()
        marker.markers["frame"].scale = (0.08, 0.08, 0.08)
        marker.prim_path = "/Visuals/DrawerFrames"
        self.scene.drawer_frames = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Cabinet/base",
            debug_vis=False,
            visualizer_cfg=marker,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Cabinet/link_0", name="top_drawer_handle", offset=OffsetCfg()
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Cabinet/link_2", name="middle_drawer_handle", offset=OffsetCfg()
                ),
            ],
        )

        # install the selected-drawer MDP
        self.observations = ObservationsCfg()
        self.rewards = RewardsCfg()
        self.terminations = TerminationsCfg()
        self.events = EventCfg()

        # shorter episodes for the drawer task
        self.episode_length_s = 8.0

        # the stack base carries an XrCfg whose default contains a lambda that breaks the rsl_rl
        # hydra cfg round-trip; this is an RL task, not XR teleop, so drop it.
        self.xr = None


@configclass
class FrankaCustomDrawerSelectedEnvCfg_PLAY(FrankaCustomDrawerSelectedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
