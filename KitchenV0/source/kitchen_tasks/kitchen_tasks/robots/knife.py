# Copyright (c) 2026. KitchenV0 project.
# SPDX-License-Identifier: BSD-3-Clause

"""PartNet knife config loaded from a local URDF."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from kitchen_tasks.assets_paths import KITCHEN_ROOT, KNIFE_URDF

KNIFE_SCALE = 0.28
KNIFE_BLADE_JOINT = "joint_0"

KITCHEN_KNIFE_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=KNIFE_URDF,
        usd_dir=str(KITCHEN_ROOT / ".generated_usd" / "knife_101054"),
        usd_file_name="knife_101054.usd",
        force_usd_conversion=False,
        fix_base=False,
        merge_fixed_joints=False,
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            target_type="position",
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=10.0, damping=1.0),
        ),
        scale=(KNIFE_SCALE, KNIFE_SCALE, KNIFE_SCALE),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=2.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.48, 0.0, 0.12),
        rot=(0.7071, 0.0, 0.0, 0.7071),
        joint_pos={KNIFE_BLADE_JOINT: 0.0},
    ),
    actuators={
        "blade": ImplicitActuatorCfg(
            joint_names_expr=[KNIFE_BLADE_JOINT],
            effort_limit_sim=10.0,
            stiffness=10.0,
            damping=1.0,
        ),
    },
)
