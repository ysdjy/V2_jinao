# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Scene extension of the official Isaac Lab Franka cabinet task.

The robot, cabinet, end-effector frame, cabinet frame, plane, and light are inherited
from the official cabinet task. This scene only adds a knife articulation beside the
cabinet so the official state-machine target names remain unchanged.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet.cabinet_env_cfg import (
    FRAME_MARKER_SMALL_CFG,
    CabinetSceneCfg,
)

from franka_cabinet_knife_tasks.robots.knife import (
    CABINET_SIDE_KNIFE_CFG,
    KNIFE_BODY_LINK,
    KNIFE_GRASP_OFFSET,
    KNIFE_HANDLE_PROXY_OFFSET,
    KNIFE_HANDLE_PROXY_SIZE,
)

HANDLE_PHYSICS_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=2.0,
    dynamic_friction=2.0,
    restitution=0.0,
    friction_combine_mode="max",
)

KNIFE_PAD_POS = (0.34, -0.34, 0.04)
KNIFE_PAD_SIZE = (0.28, 0.20, 0.08)


@configclass
class CabinetKnifeSceneCfg(CabinetSceneCfg):
    """Official cabinet scene with an additional knife prop."""

    knife_pad = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/KnifePad",
        init_state=AssetBaseCfg.InitialStateCfg(pos=KNIFE_PAD_POS),
        spawn=sim_utils.CuboidCfg(
            size=KNIFE_PAD_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38)),
        ),
    )

    knife: ArticulationCfg = CABINET_SIDE_KNIFE_CFG.replace(prim_path="{ENV_REGEX_NS}/Knife")

    knife_handle_proxy = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Knife/" + KNIFE_BODY_LINK + "/HandleProxy",
        init_state=AssetBaseCfg.InitialStateCfg(pos=KNIFE_HANDLE_PROXY_OFFSET),
        spawn=sim_utils.CuboidCfg(
            size=KNIFE_HANDLE_PROXY_SIZE,
            visible=False,
            physics_material=HANDLE_PHYSICS_MATERIAL,
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.004, rest_offset=0.0),
        ),
    )

    knife_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Knife/" + KNIFE_BODY_LINK,
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/KnifeFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Knife/" + KNIFE_BODY_LINK,
                name="knife_grasp",
                offset=OffsetCfg(pos=KNIFE_GRASP_OFFSET, rot=(0.7071, 0.0, 0.7071, 0.0)),
            ),
        ],
    )
