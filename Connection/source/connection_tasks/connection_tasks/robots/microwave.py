# Copyright (c) 2026. Connection project.
# SPDX-License-Identifier: BSD-3-Clause

"""Microwave articulation config (PartNet-Mobility 7320)."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from connection_tasks.assets_paths import MICROWAVE_7320_USD

MICROWAVE_SCALE = 0.42

MICROWAVE_BODY_LINK = "link_1"
MICROWAVE_DOOR_LINK = "link_0"
MICROWAVE_DOOR_JOINT = "joint_0"

_HANDLE_IN_DOOR_LOCAL_UNSCALED = (1.24463, -0.53513, 0.03613)
_HANDLE_SIZE_UNSCALED = (0.06986, 0.66619, 0.10427)

MICROWAVE_HANDLE_OFFSET = tuple(MICROWAVE_SCALE * c for c in _HANDLE_IN_DOOR_LOCAL_UNSCALED)
MICROWAVE_HANDLE_PROXY_OFFSET = _HANDLE_IN_DOOR_LOCAL_UNSCALED
MICROWAVE_HANDLE_PROXY_SIZE = (0.12, 0.74, 0.22)

CONNECTION_MICROWAVE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=MICROWAVE_7320_USD,
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
        mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.42, 0.0, 0.36),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={MICROWAVE_DOOR_JOINT: 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "door": ImplicitActuatorCfg(
            joint_names_expr=[MICROWAVE_DOOR_JOINT],
            effort_limit_sim=0.0,
            velocity_limit_sim=100.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
"""Microwave with a free-swinging hinged door (local USD)."""
