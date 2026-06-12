# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical calibrated geometry for the microwave revolute door.

Single source of truth (mirrors custom_drawer_config.py for the cabinet) so the env cfg (this
package) and the skill runtime (projects/franka_skill_state_machine/runtime, which re-exports this)
read the SAME numbers. Calibrated with the project's debug_microwave_door_calib.py at the deployed
microwave pose (stack_joint_pos_env_cfg.py: prim /Microwave, yaw -90deg, scale 0.35).

Facts: the door is articulation body ``link_0`` driven by revolute ``joint_0``; link_0's body origin
is the hinge and the world hinge axis is +Z (vertical swing door); the graspable free edge sits at a
constant world-metric offset in link_0's body frame.

ASSET CAVEAT: this microwave's door+body collision hulls overlap, so the door physically opens only
~10deg before contact stops it (like the locked bottom drawer). The skill is general/correct; full
opening needs the collision fixed or a different asset.

PLACEMENT CAVEAT: at the current microwave pose (world ~(0.67,-0.61), yaw -90deg) the door handle is
at world ~(0.48,-0.44,0.13), which is ~7cm beyond the Franka's comfortable -Y reach (the arm reaches
a natural pose but stalls ~0.08m short at MOVE_TO_PRE_GRASP). Move the microwave closer / more central
in the SceneLayoutModule (e.g. nearer +Y and -X) so the handle is reachable, then re-sync the cfg.
"""

from __future__ import annotations

MICROWAVE_SCALE = 0.35
DOOR_LINK = "link_0"
HINGE_JOINT = "joint_0"

# Handle (door free edge, mid height) offset in link_0's body frame, WORLD-METRIC (used by the obs
# adapter via combine_frame_transforms(link0_pos_w, link0_quat_w, this)).
HANDLE_OFFSET_LOCAL = (0.4571, -0.1843, 0.0309)

# Same offset / scale for the collision-proxy child prim (its authored local translate is multiplied
# by the parent /Microwave scale at spawn).
HANDLE_PROXY_LOCAL_OFFSET = tuple(v / MICROWAVE_SCALE for v in HANDLE_OFFSET_LOCAL)

# Thin vertical handle bar; world size ~0.045 x 0.045 x 0.18 m. Authored size / scale so spawned
# world size matches. The bar runs along the door's vertical free edge.
HANDLE_PROXY_WORLD_SIZE = (0.045, 0.045, 0.18)
HANDLE_PROXY_SIZE = tuple(v / MICROWAVE_SCALE for v in HANDLE_PROXY_WORLD_SIZE)

# Door-open success angle (rad). Modest because the asset caps physical open; raise after a fix.
OPEN_SUCCESS_ANGLE = 0.14   # ~8 deg
CLOSE_SUCCESS_ANGLE = 0.03  # ~1.7 deg

DOOR_TARGETS = {
    "microwave": {
        "asset_name": "microwave",
        "joint_name": HINGE_JOINT,
        "link_name": DOOR_LINK,
        "handle_offset": HANDLE_OFFSET_LOCAL,
        "open_success_angle": OPEN_SUCCESS_ANGLE,
        "close_success_angle": CLOSE_SUCCESS_ANGLE,
    },
}
