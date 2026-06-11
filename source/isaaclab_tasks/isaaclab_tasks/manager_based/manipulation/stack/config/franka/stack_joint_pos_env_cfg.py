# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from isaaclab_tasks.manager_based.manipulation.stack.stack_env_cfg import StackEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


KNIFE_BODY_LINK = "base"
KNIFE_HANDLE_PROXY_OFFSET = (0.18304, 0.01413, -0.02520)
KNIFE_HANDLE_PROXY_SIZE = (0.98, 0.24, 0.20)
CABINET_BOTTOM_DRAWER_LINK = "link_1"
CABINET_BOTTOM_HANDLE_PROXY_OFFSET = (0.11946, 0.01491, 1.06183)
# Thin handle-bar collision proxy (link-local size; world size = this * cabinet scale 0.62).
# Long axis (local X) halved per request: (0.10, 0.028, 0.028) -> world ~(0.062, 0.017, 0.017) m.
CABINET_BOTTOM_HANDLE_PROXY_SIZE = (0.10, 0.028, 0.028)

# Top/middle drawer handle proxies (visible collision boxes), mirroring the bottom one.
# Position = calibrated handle offset (link BODY frame, from debug_drawer_handle_calib.py) divided by
# the cabinet USD scale, because the proxy is a child of the link prim (pre-scale local frame).
from .custom_drawer_config import HANDLE_LOCAL_OFFSET as _HANDLE_OFFSET  # noqa: E402

CABINET_SCALE = 0.62
CABINET_TOP_DRAWER_LINK = "link_0"
CABINET_MIDDLE_DRAWER_LINK = "link_2"
CABINET_TOP_HANDLE_PROXY_OFFSET = tuple(v / CABINET_SCALE for v in _HANDLE_OFFSET["top_drawer"])
CABINET_MIDDLE_HANDLE_PROXY_OFFSET = tuple(v / CABINET_SCALE for v in _HANDLE_OFFSET["middle_drawer"])
# same slim handle-bar size for all three drawers (see CABINET_BOTTOM_HANDLE_PROXY_SIZE note)
CABINET_HANDLE_PROXY_SIZE = CABINET_BOTTOM_HANDLE_PROXY_SIZE


def _repo_path(relative_path: str) -> str:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / relative_path
        if candidate.exists():
            return str(candidate)
    return str(Path.cwd() / relative_path)


def _simv2_usd_path(relative_path: str) -> str:
    """Resolve assets copied under the repository's simv2/USD directory."""
    return _repo_path(f"simv2/USD/{relative_path}")


@configclass
class EventCfg:
    """Configuration for events."""

    init_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400],
        },
    )

    randomize_franka_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    randomize_cube_positions = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.4, 0.6), "y": (-0.10, 0.10), "z": (0.0203, 0.0203), "yaw": (-1.0, 1, 0)},
            "min_separation": 0.1,
            "asset_cfgs": [SceneEntityCfg("cube_1"), SceneEntityCfg("cube_2"), SceneEntityCfg("cube_3")],
        },
    )


@configclass
class FrankaCubeStackEnvCfg(StackEnvCfg):
    """Configuration for the Franka Cube Stack Environment."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set events
        self.events = EventCfg()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.usd_path = _repo_path(
            "Connection/assets/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd"
        )
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Remove the table and place the robot/cubes directly on the ground plane.
        self.scene.table = None
        self.scene.plane.init_state.pos = [0.0, 0.0, -0.025]
        self.scene.plane.spawn = sim_utils.CuboidCfg(
            size=(10.0, 10.0, 0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.02, 0.05, 0.11)),
        )

        # Add semantics to ground
        self.scene.plane.semantic_tags = [("class", "ground")]

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        # utilities for gripper status check
        self.gripper_joint_names = ["panda_finger_.*"]
        self.gripper_open_val = 0.04
        self.gripper_threshold = 0.005

        # Rigid body properties of each cube
        cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )
        cube_collision_properties = sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0)
        cube_mass_properties = sim_utils.MassPropertiesCfg(mass=0.08)
        cube_size = (0.0406, 0.0406, 0.0406)

        # Set each stacking cube deterministically
        self.scene.cube_1 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_1",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, 0.0203], rot=[1, 0, 0, 0]),
            spawn=sim_utils.CuboidCfg(
                size=cube_size,
                rigid_props=cube_properties,
                mass_props=cube_mass_properties,
                collision_props=cube_collision_properties,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.25, 0.9)),
                semantic_tags=[("class", "cube_1")],
            ),
        )
        self.scene.cube_2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_2",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.55, 0.05, 0.0203], rot=[1, 0, 0, 0]),
            spawn=sim_utils.CuboidCfg(
                size=cube_size,
                rigid_props=cube_properties,
                mass_props=cube_mass_properties,
                collision_props=cube_collision_properties,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.08)),
                semantic_tags=[("class", "cube_2")],
            ),
        )
        self.scene.cube_3 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_3",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.60, -0.1, 0.0203], rot=[1, 0, 0, 0]),
            spawn=sim_utils.CuboidCfg(
                size=cube_size,
                rigid_props=cube_properties,
                mass_props=cube_mass_properties,
                collision_props=cube_collision_properties,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.65, 0.2)),
                semantic_tags=[("class", "cube_3")],
            ),
        )

        self.scene.cabinet = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Cabinet",
            spawn=UsdFileCfg(
                usd_path=_simv2_usd_path("Cabinet_44853/cabinet.usd"),
                scale=(0.62, 0.62, 0.62),
                activate_contact_sensors=False,
                rigid_props=RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=5.0),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=12,
                    solver_velocity_iteration_count=1,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=8.0),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                semantic_tags=[("class", "cabinet")],
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                # Matches the user's fixed scene (saved_scenes/v0_layout/scene_v0_*.usd): cabinet placed
                # within the arm's workspace, yaw +90deg about Z so the drawers open toward -Y (toward
                # the robot) and the handles are reachable.
                pos=(0.27402, 0.91583, 0.323),
                rot=(0.70710678, 0.0, 0.0, 0.70710678),
                joint_pos={".*": 0.0},
                joint_vel={".*": 0.0},
            ),
            actuators={
                "drawers": ImplicitActuatorCfg(
                    joint_names_expr=["joint_0", "joint_1", "joint_2"],
                    effort_limit_sim=87.0,
                    velocity_limit_sim=100.0,
                    stiffness=10.0,
                    damping=1.0,
                ),
            },
        )
        self.scene.cabinet_handle_proxy = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Cabinet/" + CABINET_BOTTOM_DRAWER_LINK + "/BottomHandleProxy",
            init_state=AssetBaseCfg.InitialStateCfg(pos=CABINET_BOTTOM_HANDLE_PROXY_OFFSET),
            spawn=sim_utils.CuboidCfg(
                size=CABINET_BOTTOM_HANDLE_PROXY_SIZE,
                visible=True,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                    friction_combine_mode="max",
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.9, 1.0), opacity=0.35),
            ),
        )

        # visible collision proxies for the TOP and MIDDLE drawer handles (bottom already has one)
        def _handle_proxy(link: str, name: str, offset, color):
            return AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Cabinet/" + link + "/" + name,
                init_state=AssetBaseCfg.InitialStateCfg(pos=offset),
                spawn=sim_utils.CuboidCfg(
                    size=CABINET_HANDLE_PROXY_SIZE,
                    visible=True,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=1.0, dynamic_friction=1.0, restitution=0.0, friction_combine_mode="max"
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, opacity=0.35),
                ),
            )

        self.scene.cabinet_top_handle_proxy = _handle_proxy(
            CABINET_TOP_DRAWER_LINK, "TopHandleProxy", CABINET_TOP_HANDLE_PROXY_OFFSET, (1.0, 0.4, 0.05)
        )
        self.scene.cabinet_middle_handle_proxy = _handle_proxy(
            CABINET_MIDDLE_DRAWER_LINK, "MiddleHandleProxy", CABINET_MIDDLE_HANDLE_PROXY_OFFSET, (0.4, 1.0, 0.05)
        )

        self.scene.knife = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Knife",
            spawn=UsdFileCfg(
                usd_path=_simv2_usd_path("Knife_101054/knife.usd"),
                scale=(0.12, 0.12, 0.12),
                activate_contact_sensors=False,
                rigid_props=RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=2.0),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=12,
                    solver_velocity_iteration_count=1,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.06),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.004, rest_offset=0.0),
                semantic_tags=[("class", "knife")],
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.35, 0.28, 0.095),
                rot=(0.7071, 0.0, 0.0, 0.7071),
                joint_pos={"joint_0": -0.2},
                joint_vel={".*": 0.0},
            ),
            actuators={
                "blade_lock": ImplicitActuatorCfg(
                    joint_names_expr=["joint_0"],
                    effort_limit_sim=0.0,
                    velocity_limit_sim=20.0,
                    stiffness=0.0,
                    damping=0.0,
                ),
            },
        )
        self.scene.knife_handle_proxy = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Knife/" + KNIFE_BODY_LINK + "/HandleProxy",
            init_state=AssetBaseCfg.InitialStateCfg(pos=KNIFE_HANDLE_PROXY_OFFSET),
            spawn=sim_utils.CuboidCfg(
                size=KNIFE_HANDLE_PROXY_SIZE,
                visible=True,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=4.0,
                    dynamic_friction=4.0,
                    restitution=0.0,
                    friction_combine_mode="max",
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.004, rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.75, 0.05), opacity=0.35),
            ),
        )

        # Coffee machine prop — matches the user's fixed scene (saved_scenes/v0_layout): a fixed-base
        # articulation placed on the -Y side, yaw -90deg about Z, scale 0.2. It is a scene
        # obstacle/prop (not a manipulation target); its joints are held by a soft implicit actuator.
        self.scene.coffee_machine = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/CoffeeMachine",
            spawn=UsdFileCfg(
                usd_path=_repo_path("SapienAssetPipeline/usd_assets/CoffeeMachine_103046/coffeemachine.usd"),
                scale=(0.2, 0.2, 0.2),
                activate_contact_sensors=False,
                rigid_props=RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=5.0),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=12,
                    solver_velocity_iteration_count=1,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                semantic_tags=[("class", "coffee_machine")],
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.43151966, -0.48372657, 0.135),
                rot=(0.70710678, 0.0, 0.0, -0.70710678),
                joint_pos={".*": 0.0},
                joint_vel={".*": 0.0},
            ),
            actuators={
                "all_joints": ImplicitActuatorCfg(
                    joint_names_expr=["joint_.*"],
                    effort_limit_sim=50.0,
                    velocity_limit_sim=100.0,
                    stiffness=100.0,
                    damping=10.0,
                ),
            },
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )
