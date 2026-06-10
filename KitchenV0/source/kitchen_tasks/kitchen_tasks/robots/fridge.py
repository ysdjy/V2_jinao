# Copyright (c) 2026. KitchenV0 project.
# SPDX-License-Identifier: BSD-3-Clause

"""PartNet refrigerator config."""

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from kitchen_tasks.assets_paths import FRIDGE_USD

FRIDGE_SCALE = 0.65
FRIDGE_DOOR_JOINT = "joint_0"
FRIDGE_BODY_LINK = "link_1"
FRIDGE_DOOR_LINK = "link_0"
FRIDGE_CLOSED_POS = 0.0
FRIDGE_OPEN_15_DEG = math.radians(15.0)
FRIDGE_OPEN_45_DEG = math.radians(45.0)

_HANDLE_IN_DOOR_LOCAL_UNSCALED = (-1.01125, -0.55402, 0.04661)
FRIDGE_HANDLE_OFFSET = tuple(c * FRIDGE_SCALE for c in _HANDLE_IN_DOOR_LOCAL_UNSCALED)

KITCHEN_FRIDGE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=FRIDGE_USD,
        scale=(FRIDGE_SCALE, FRIDGE_SCALE, FRIDGE_SCALE),
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
        # Proven open-fridge layout from the previous Connection prototype:
        # the handle starts inside the fixed-base Franka workspace, while the
        # body is far enough to avoid collisions during approach.
        pos=(0.63, -0.52, 0.43),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={FRIDGE_DOOR_JOINT: FRIDGE_CLOSED_POS},
    ),
    actuators={
        "door": ImplicitActuatorCfg(
            joint_names_expr=[FRIDGE_DOOR_JOINT],
            effort_limit_sim=87.0,
            velocity_limit_sim=100.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)

FRANKA_FRIDGE_READY_QPOS = {
    "panda_joint1": 0.4283,
    "panda_joint2": -1.6981,
    "panda_joint3": 0.0475,
    "panda_joint4": -3.0493,
    "panda_joint5": -1.0821,
    "panda_joint6": 2.6595,
    "panda_joint7": -1.2743,
    "panda_finger_joint.*": 0.04,
}
"""Franka ready pose with the gripper already oriented toward the fridge handle."""
