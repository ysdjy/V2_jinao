"""Convert a canonical delta-EE Action into an IsaacLab action vector.

Target env families:
  * Isaac-*-Franka-IK-Rel-v0  -> action = [dx,dy,dz, drx,dry,drz, gripper]  (7D)
        position delta in meters, rotation delta as axis-angle (rad),
        gripper as a single scalar (env binarizes: >0 open, <=0 close, or [-1,1]).
  * Isaac-*-Franka-IK-Abs-v0  -> action = [x,y,z, qw,qx,qy,qz, gripper] (8D)
        absolute pose; we integrate the delta onto the current EE pose.

Returns a plain python list (the IsaacLab caller wraps it into a torch tensor).
numpy is used only if available; falls back to math for quaternion ops.
"""

from __future__ import annotations

import math
from typing import Any, Optional


def _quat_mul(a, b):
    # a,b in [w,x,y,z]
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def _axis_angle_to_quat_wxyz(rot):
    angle = math.sqrt(sum(c * c for c in rot))
    if angle < 1e-8:
        return [1.0, 0.0, 0.0, 0.0]
    axis = [c / angle for c in rot]
    s = math.sin(angle / 2.0)
    return [math.cos(angle / 2.0), axis[0] * s, axis[1] * s, axis[2] * s]


def delta_to_ik_rel(action: dict[str, Any]) -> list[float]:
    """7D relative IK action. Rotation kept as axis-angle (what IK-Rel expects)."""
    dpos = list(action.get("delta_ee_position", [0, 0, 0]))[:3]
    drot = list(action.get("delta_ee_rot", [0, 0, 0]))[:3]
    while len(dpos) < 3:
        dpos.append(0.0)
    while len(drot) < 3:
        drot.append(0.0)
    grip = float(action.get("gripper", 0.0))
    return [float(v) for v in dpos] + [float(v) for v in drot] + [grip]


def delta_to_ik_abs(
    action: dict[str, Any],
    current_ee_position: list[float],
    current_ee_quat_xyzw: list[float],
) -> list[float]:
    """8D absolute IK action [x,y,z, qw,qx,qy,qz, gripper] integrated from current pose."""
    dpos = list(action.get("delta_ee_position", [0, 0, 0]))[:3]
    drot = list(action.get("delta_ee_rot", [0, 0, 0]))[:3]
    grip = float(action.get("gripper", 0.0))

    new_pos = [float(current_ee_position[i]) + float(dpos[i]) for i in range(3)]
    # current quat xyzw -> wxyz
    cx, cy, cz, cw = current_ee_quat_xyzw
    cur_wxyz = [cw, cx, cy, cz]
    dq = _axis_angle_to_quat_wxyz(drot)
    new_wxyz = _quat_mul(dq, cur_wxyz)
    # normalize
    n = math.sqrt(sum(c * c for c in new_wxyz)) or 1.0
    new_wxyz = [c / n for c in new_wxyz]
    return new_pos + new_wxyz + [grip]


def build_isaac_action(
    action: dict[str, Any],
    env_kind: str,
    current_ee_position: Optional[list[float]] = None,
    current_ee_quat_xyzw: Optional[list[float]] = None,
) -> list[float]:
    """env_kind in {"ik_rel","ik_abs"}."""
    if env_kind == "ik_rel":
        return delta_to_ik_rel(action)
    if env_kind == "ik_abs":
        if current_ee_position is None or current_ee_quat_xyzw is None:
            raise ValueError("ik_abs needs current EE pose")
        return delta_to_ik_abs(action, current_ee_position, current_ee_quat_xyzw)
    raise ValueError(f"unknown env_kind {env_kind}")
