# Copyright (c) 2026. KitchenV0 project.
# SPDX-License-Identifier: BSD-3-Clause

"""PartNet microwave config loaded from a local URDF."""

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from kitchen_tasks.assets_paths import KITCHEN_ROOT, MICROWAVE_URDF

MICROWAVE_SCALE = 0.55
MICROWAVE_DOOR_JOINT = "joint_0"
MICROWAVE_BODY_LINK = "link_1"
MICROWAVE_DOOR_LINK = "link_0"
MICROWAVE_CLOSED_POS = 0.0
MICROWAVE_OPEN_45_DEG = math.radians(45.0)

KITCHEN_MICROWAVE_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=MICROWAVE_URDF,
        usd_dir=str(KITCHEN_ROOT / ".generated_usd" / "microwave_7320"),
        usd_file_name="microwave_7320.usd",
        force_usd_conversion=False,
        fix_base=True,
        merge_fixed_joints=False,
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            target_type="none",
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
        scale=(MICROWAVE_SCALE, MICROWAVE_SCALE, MICROWAVE_SCALE),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.58, -0.58, 0.32),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={MICROWAVE_DOOR_JOINT: MICROWAVE_CLOSED_POS},
    ),
    actuators={
        "door": ImplicitActuatorCfg(
            joint_names_expr=[MICROWAVE_DOOR_JOINT],
            effort_limit_sim=30.0,
            velocity_limit_sim=100.0,
            stiffness=0.0,
            damping=1.5,
        ),
    },
)
