# Copyright (c) 2026. Connection project.
# SPDX-License-Identifier: BSD-3-Clause

"""Three-drawer cabinet config (PartNet-Mobility 44853).

The cabinet has three prismatic drawer joints:

* ``joint_0`` / ``link_0``: top drawer
* ``joint_2`` / ``link_2``: middle drawer
* ``joint_1`` / ``link_1``: bottom drawer

The fixed joint in the URDF rotates the PartNet frame into the Isaac frame, so the
bottom drawer is not the numerically last link. The task intentionally targets
``joint_1`` and the handle on ``link_1``.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from connection_tasks.assets_paths import CABINET_44853_USD

CABINET_SCALE = 0.62

CABINET_BODY_LINK = "link_3"
CABINET_BOTTOM_DRAWER_LINK = "link_1"
CABINET_BOTTOM_DRAWER_JOINT = "joint_1"
CABINET_DRAWER_JOINTS = ("joint_0", "joint_1", "joint_2")

_BOTTOM_HANDLE_IN_LINK_LOCAL_UNSCALED = (0.11946, 0.01491, 1.06183)
_BOTTOM_HANDLE_SIZE_UNSCALED = (0.18994, 0.02568, 0.04968)

CABINET_BOTTOM_HANDLE_OFFSET = tuple(CABINET_SCALE * c for c in _BOTTOM_HANDLE_IN_LINK_LOCAL_UNSCALED)
CABINET_BOTTOM_HANDLE_PROXY_OFFSET = _BOTTOM_HANDLE_IN_LINK_LOCAL_UNSCALED
CABINET_BOTTOM_HANDLE_PROXY_SIZE = (0.24, 0.08, 0.10)

CONNECTION_CABINET_44853_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=CABINET_44853_USD,
        scale=(CABINET_SCALE, CABINET_SCALE, CABINET_SCALE),
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
        mass_props=sim_utils.MassPropertiesCfg(mass=8.0),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.48, 0.0, 0.58),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "drawers": ImplicitActuatorCfg(
            joint_names_expr=list(CABINET_DRAWER_JOINTS),
            effort_limit_sim=87.0,
            velocity_limit_sim=100.0,
            stiffness=0.0,
            damping=1.0,
        ),
    },
)
"""Three-drawer cabinet with free-sliding drawers (local USD)."""
