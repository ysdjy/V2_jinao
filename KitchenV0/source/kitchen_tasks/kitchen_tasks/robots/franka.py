# Copyright (c) 2026. KitchenV0 project.
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Emika Panda configs backed by clone-relative local assets."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from kitchen_tasks.assets_paths import FRANKA_PANDA_USD

KITCHEN_FRANKA_READY_QPOS = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.569,
    "panda_joint3": 0.0,
    "panda_joint4": -2.810,
    "panda_joint5": 0.0,
    "panda_joint6": 3.037,
    "panda_joint7": 0.741,
    "panda_finger_joint.*": 0.04,
}

KITCHEN_FRANKA_PANDA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=FRANKA_PANDA_USD,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(joint_pos=KITCHEN_FRANKA_READY_QPOS),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit_sim=200.0,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

KITCHEN_FRANKA_PANDA_HIGH_PD_CFG = KITCHEN_FRANKA_PANDA_CFG.copy()
KITCHEN_FRANKA_PANDA_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
KITCHEN_FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].stiffness = 400.0
KITCHEN_FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].damping = 80.0
KITCHEN_FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 400.0
KITCHEN_FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].damping = 80.0
