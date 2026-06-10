# Copyright (c) 2026. Connection project.
# SPDX-License-Identifier: BSD-3-Clause

"""Knife config (PartNet-Mobility 101054).

The converted asset remains a small articulation because the blade has a revolute
joint. The blade joint is passive so all non-robot motion in the task comes from
physics contact.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from connection_tasks.assets_paths import KNIFE_101054_USD

KNIFE_SCALE = 0.12
KNIFE_BLADE_JOINT = "joint_0"
KNIFE_BODY_LINK = "base"
KNIFE_GRASP_OFFSET = (0.0, 0.0, 0.045)
KNIFE_HANDLE_PROXY_OFFSET = (0.18304, 0.01413, -0.02520)
KNIFE_HANDLE_PROXY_SIZE = (0.98, 0.24, 0.20)

CONNECTION_KNIFE_CFG = ArticulationCfg(
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
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.004, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.38, 0.0, 0.10),
        rot=(0.7071, 0.0, 0.0, 0.7071),
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
"""Small dynamic knife articulation (local USD)."""
