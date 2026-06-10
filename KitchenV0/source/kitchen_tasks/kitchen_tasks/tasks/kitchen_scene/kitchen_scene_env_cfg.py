# Copyright (c) 2026. KitchenV0 project.
# SPDX-License-Identifier: BSD-3-Clause

"""KitchenV0 scene configuration.

Robot frame convention: the Franka base is at the origin and faces +x.

- left of robot: +y, refrigerator
- right of robot: -y, microwave
- behind robot: -x, cabinet
- in front of robot: +x, knife
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass

from kitchen_tasks.assets_paths import SEKTION_CABINET_USD
from kitchen_tasks.robots.fridge import (
    FRIDGE_BODY_LINK,
    FRIDGE_DOOR_JOINT,
    FRIDGE_DOOR_LINK,
    FRIDGE_HANDLE_OFFSET,
    FRIDGE_OPEN_15_DEG,
    FRIDGE_OPEN_45_DEG,
    KITCHEN_FRIDGE_CFG,
)
from kitchen_tasks.robots.knife import KITCHEN_KNIFE_CFG
from kitchen_tasks.robots.microwave import (
    KITCHEN_MICROWAVE_CFG,
    MICROWAVE_BODY_LINK,
    MICROWAVE_DOOR_JOINT,
    MICROWAVE_DOOR_LINK,
    MICROWAVE_OPEN_45_DEG,
)

from . import mdp

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.08, 0.08, 0.08)

CABINET_BOTTOM_DRAWER_JOINT = "drawer_bottom_joint"
CABINET_BOTTOM_DRAWER_OPEN_POS = 0.32

CABINET_POS = (-0.72, 0.0, 0.40)
CABINET_ROT = (1.0, 0.0, 0.0, 0.0)

MICROWAVE_STAND_POS = (0.58, -0.58, 0.13)
MICROWAVE_STAND_SIZE = (0.62, 0.46, 0.26)
KNIFE_TABLE_POS = (0.48, 0.0, 0.05)
KNIFE_TABLE_SIZE = (0.34, 0.26, 0.10)


@configclass
class KitchenSceneCfg(InteractiveSceneCfg):
    """Scene with Franka, fridge, microwave, cabinet and knife."""

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    fridge: ArticulationCfg = KITCHEN_FRIDGE_CFG.replace(prim_path="{ENV_REGEX_NS}/Fridge")

    microwave_stand = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/MicrowaveStand",
        init_state=AssetBaseCfg.InitialStateCfg(pos=MICROWAVE_STAND_POS),
        spawn=sim_utils.CuboidCfg(
            size=MICROWAVE_STAND_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.28, 0.29, 0.31)),
        ),
    )

    microwave: ArticulationCfg = KITCHEN_MICROWAVE_CFG.replace(prim_path="{ENV_REGEX_NS}/Microwave")

    cabinet = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet",
        spawn=sim_utils.UsdFileCfg(usd_path=SEKTION_CABINET_USD, activate_contact_sensors=False),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=CABINET_POS,
            rot=CABINET_ROT,
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                effort_limit_sim=87.0,
                stiffness=10.0,
                damping=1.0,
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit_sim=87.0,
                stiffness=10.0,
                damping=2.5,
            ),
        },
    )

    knife_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/KnifeTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=KNIFE_TABLE_POS),
        spawn=sim_utils.CuboidCfg(
            size=KNIFE_TABLE_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.42, 0.38, 0.32)),
        ),
    )

    knife: ArticulationCfg = KITCHEN_KNIFE_CFG.replace(prim_path="{ENV_REGEX_NS}/Knife")

    fridge_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Fridge/" + FRIDGE_BODY_LINK,
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/KitchenFridgeFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Fridge/" + FRIDGE_DOOR_LINK,
                name="fridge_handle",
                offset=OffsetCfg(pos=FRIDGE_HANDLE_OFFSET, rot=(0.5, 0.5, -0.5, -0.5)),
            ),
        ],
    )

    microwave_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Microwave/" + MICROWAVE_BODY_LINK,
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/KitchenMicrowaveFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Microwave/" + MICROWAVE_DOOR_LINK,
                name="microwave_handle",
                offset=OffsetCfg(pos=(-0.12, -0.18, 0.02), rot=(0.5, 0.5, -0.5, -0.5)),
            ),
        ],
    )

    cabinet_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet/sektion",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/KitchenCabinetFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Cabinet/drawer_handle_bottom",
                name="bottom_drawer_handle",
                offset=OffsetCfg(pos=(0.305, 0.0, 0.01), rot=(0.5, 0.5, -0.5, -0.5)),
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
    """Action specifications set by the robot-specific config."""

    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Minimal observations for zero-action and future state-machine bring-up."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """No reward terms in v0."""


@configclass
class EventCfg:
    """Configuration for reset events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class TerminationsCfg:
    """Episode termination terms."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class KitchenSceneEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for KitchenV0."""

    scene: KitchenSceneCfg = KitchenSceneCfg(num_envs=128, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 1
        self.episode_length_s = 40.0
        self.viewer.eye = (1.55, -1.75, 1.85)
        self.viewer.lookat = (0.05, 0.0, 0.35)
        self.sim.dt = 1 / 60
        self.sim.render_interval = self.decimation
