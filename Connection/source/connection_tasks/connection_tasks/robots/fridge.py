# Copyright (c) 2026. Connection project.
# SPDX-License-Identifier: BSD-3-Clause

"""Refrigerator articulation config (PartNet-Mobility 12252) backed by local USD.

Structure of the converted USD (see Connection/tools/inspect_fridge_usd.py):
  * default prim ``/partnet_...`` -> spawned under ``{ENV}/Fridge``
  * ``base`` (fixed to world via ``root_joint`` because of --fix-base)
  * ``link_1`` : fridge body (parent of the door hinge)
  * ``link_0`` : the door (carries the vertical handle)
  * ``joint_0`` : revolute door hinge, range 0 .. 180 deg  (this is what we open)
  * ``joint_1`` : fixed joint base->link_1

The asset is scaled down (FRIDGE_SCALE) so the door handle lands inside a
table-top Franka workspace.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from connection_tasks.assets_paths import FRIDGE_USD

# uniform scale applied when spawning the fridge
FRIDGE_SCALE = 0.65

# name of the revolute door joint inside the USD
FRIDGE_DOOR_JOINT = "joint_0"

# door prim (relative to the fridge root prim) that carries the handle
FRIDGE_DOOR_LINK = "link_0"
FRIDGE_BODY_LINK = "link_1"

# handle center expressed in the door (link_0) local frame, UNSCALED.
# Computed by inspect_fridge_usd.py (HANDLE_IN_LINK0_LOCAL).
_HANDLE_IN_DOOR_LOCAL_UNSCALED = (-1.01125, -0.55402, 0.04661)

# Franka "ready" joint configuration with the gripper already pointing horizontally
# forward, just in front of the door handle. Used as the task init pose so the arm
# never has to flip from down->forward (which would sweep into the door).
# Found via Connection/tools/probe_reach.py.
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

# same offset after applying FRIDGE_SCALE -> use directly as a FrameTransformer offset
FRIDGE_HANDLE_OFFSET = tuple(c * FRIDGE_SCALE for c in _HANDLE_IN_DOOR_LOCAL_UNSCALED)
FRIDGE_HANDLE_PROXY_OFFSET = _HANDLE_IN_DOOR_LOCAL_UNSCALED
FRIDGE_HANDLE_PROXY_SIZE = (0.10, 1.00, 0.22)

CONNECTION_FRIDGE_CFG = ArticulationCfg(
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
        # overridden by the environment config (placed in front of the robot)
        # handle lands near (0.40, 0.10, 0.55) in env frame when door is closed
        pos=(0.63, -0.12, 0.43),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={FRIDGE_DOOR_JOINT: 0.0},
    ),
    actuators={
        # Passive hinge: no drive, no scripted force. The state machine can only
        # move the door through gripper contact with the handle collision proxy.
        "door": ImplicitActuatorCfg(
            joint_names_expr=[FRIDGE_DOOR_JOINT],
            effort_limit_sim=0.0,
            velocity_limit_sim=100.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
"""Refrigerator (PartNet 12252) as a fixed-base articulation with a revolute door."""
