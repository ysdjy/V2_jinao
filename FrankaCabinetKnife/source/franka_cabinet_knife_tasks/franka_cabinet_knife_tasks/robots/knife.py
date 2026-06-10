# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Knife config for the PartNet-Mobility 101054 USD already used in this workspace."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from franka_cabinet_knife_tasks.assets_paths import KNIFE_101054_USD

KNIFE_SCALE = 0.12
KNIFE_BLADE_JOINT = "joint_0"
KNIFE_BODY_LINK = "base"
KNIFE_GRASP_OFFSET = (0.0, 0.0, 0.045)
KNIFE_HANDLE_PROXY_OFFSET = (0.18304, 0.01413, -0.02520)
KNIFE_HANDLE_PROXY_SIZE = (0.98, 0.24, 0.20)
KNIFE_SIDE_OF_ROBOT_POS = (0.34, -0.34, 0.095)
KNIFE_SIDE_OF_ROBOT_ROT = (0.7071, 0.0, 0.0, 0.7071)

CABINET_SIDE_KNIFE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=KNIFE_101054_USD,
        scale=(KNIFE_SCALE, KNIFE_SCALE, KNIFE_SCALE),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=2.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=1,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.06),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=KNIFE_SIDE_OF_ROBOT_POS,
        rot=KNIFE_SIDE_OF_ROBOT_ROT,
        joint_pos={KNIFE_BLADE_JOINT: -0.2},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "blade_lock": ImplicitActuatorCfg(
            joint_names_expr=[KNIFE_BLADE_JOINT],
            effort_limit_sim=0.0,
            velocity_limit_sim=20.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
"""Small dynamic knife articulation placed beside the official cabinet."""
