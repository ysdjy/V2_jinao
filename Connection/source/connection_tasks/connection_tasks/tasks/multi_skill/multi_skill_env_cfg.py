# Copyright (c) 2026. Connection project.
# SPDX-License-Identifier: BSD-3-Clause

"""Cabinet-only manipulation scene for the Connection task.

The fridge and microwave stations are intentionally disabled while the cabinet
station is being calibrated. The active scene contains one Franka, PartNet
cabinet 44853, the bottom-drawer handle collision proxy, and the knife asset.
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

from connection_tasks.robots.cabinet_44853 import (
    CABINET_BODY_LINK,
    CABINET_BOTTOM_DRAWER_JOINT,
    CABINET_BOTTOM_DRAWER_LINK,
    CABINET_BOTTOM_HANDLE_OFFSET,
    CABINET_BOTTOM_HANDLE_PROXY_OFFSET,
    CABINET_BOTTOM_HANDLE_PROXY_SIZE,
    CONNECTION_CABINET_44853_CFG,
)
from connection_tasks.robots.knife import (
    CONNECTION_KNIFE_CFG,
    KNIFE_BODY_LINK,
    KNIFE_GRASP_OFFSET,
    KNIFE_HANDLE_PROXY_OFFSET,
    KNIFE_HANDLE_PROXY_SIZE,
)

from . import mdp

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)

HANDLE_PHYSICS_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=4.0,
    dynamic_friction=4.0,
    restitution=0.0,
    friction_combine_mode="max",
)

# Active cabinet sub-scene origin inside each Isaac Lab environment.
CABINET_SCENE_ORIGIN = (0.0, 0.0, 0.0)

# Ground the cabinet and place it on the robot's reachable side. The USD root is
# around 0.32 m above the cabinet's lowest geometry point after scaling, so that
# z places the feet on the ground plane instead of leaving the cabinet floating.
CABINET_POS_LOCAL = (0.75, -0.45, 0.323)

KNIFE_PAD_POS_LOCAL = (0.36, 0.08, 0.04)
KNIFE_PAD_SIZE = (0.26, 0.18, 0.08)
KNIFE_POS_LOCAL = (0.36, 0.08, 0.095)


def _add_pos(origin: tuple[float, float, float], pos: tuple[float, float, float]) -> tuple[float, float, float]:
    return (origin[0] + pos[0], origin[1] + pos[1], origin[2] + pos[2])


def _cabinet_cfg() -> ArticulationCfg:
    cfg = CONNECTION_CABINET_44853_CFG.replace(prim_path="{ENV_REGEX_NS}/Cabinet")
    cfg.init_state.pos = _add_pos(CABINET_SCENE_ORIGIN, CABINET_POS_LOCAL)
    return cfg


def _knife_cfg() -> ArticulationCfg:
    cfg = CONNECTION_KNIFE_CFG.replace(prim_path="{ENV_REGEX_NS}/Knife")
    cfg.init_state.pos = _add_pos(CABINET_SCENE_ORIGIN, KNIFE_POS_LOCAL)
    return cfg


##
# Scene definition
##


@configclass
class MultiSkillSceneCfg(InteractiveSceneCfg):
    """Scene containing the cabinet manipulation station."""

    # Frankas + end-effector frames are populated by the robot-specific config.
    cabinet_robot: ArticulationCfg = MISSING
    cabinet_ee_frame: FrameTransformerCfg = MISSING

    # -- cabinet bottom drawer + knife
    cabinet: ArticulationCfg = _cabinet_cfg()
    cabinet_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet/" + CABINET_BODY_LINK,
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/CabinetFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Cabinet/" + CABINET_BOTTOM_DRAWER_LINK,
                name="cabinet_bottom_drawer_handle",
                offset=OffsetCfg(pos=CABINET_BOTTOM_HANDLE_OFFSET, rot=(0.7071, 0.0, -0.7071, 0.0)),
            ),
        ],
    )
    cabinet_handle_proxy = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet/" + CABINET_BOTTOM_DRAWER_LINK + "/BottomHandleProxy",
        init_state=AssetBaseCfg.InitialStateCfg(pos=CABINET_BOTTOM_HANDLE_PROXY_OFFSET),
        spawn=sim_utils.CuboidCfg(
            size=CABINET_BOTTOM_HANDLE_PROXY_SIZE,
            visible=False,
            physics_material=HANDLE_PHYSICS_MATERIAL,
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.004, rest_offset=0.0),
        ),
    )
    knife_pad = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/KnifePad",
        init_state=AssetBaseCfg.InitialStateCfg(pos=_add_pos(CABINET_SCENE_ORIGIN, KNIFE_PAD_POS_LOCAL)),
        spawn=sim_utils.CuboidCfg(
            size=KNIFE_PAD_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38)),
        ),
    )
    knife: ArticulationCfg = _knife_cfg()
    knife_handle_proxy = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Knife/" + KNIFE_BODY_LINK + "/HandleProxy",
        init_state=AssetBaseCfg.InitialStateCfg(pos=KNIFE_HANDLE_PROXY_OFFSET),
        spawn=sim_utils.CuboidCfg(
            size=KNIFE_HANDLE_PROXY_SIZE,
            visible=False,
            physics_material=HANDLE_PHYSICS_MATERIAL,
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.004, rest_offset=0.0),
        ),
    )
    knife_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Knife/" + KNIFE_BODY_LINK,
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/KnifeFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Knife/" + KNIFE_BODY_LINK,
                name="knife_grasp",
                offset=OffsetCfg(pos=KNIFE_GRASP_OFFSET, rot=(0.7071, 0.0, 0.7071, 0.0)),
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
    """Action specifications (set by the robot-specific config)."""

    cabinet_arm_action: mdp.JointPositionActionCfg = MISSING
    cabinet_gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        cabinet_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("cabinet_robot")}
        )
        cabinet_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("cabinet_robot")}
        )
        cabinet_bottom_drawer_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=[CABINET_BOTTOM_DRAWER_JOINT])},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms (empty: the state machines drive the scene)."""


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class MultiSkillEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cabinet-only manipulation environment."""

    scene: MultiSkillSceneCfg = MultiSkillSceneCfg(num_envs=1, env_spacing=8.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 1
        self.episode_length_s = 60.0
        self.viewer.eye = (3.0, -4.0, 2.2)
        self.viewer.lookat = (0.55, 0.30, 0.45)
        self.sim.dt = 1 / 60
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
