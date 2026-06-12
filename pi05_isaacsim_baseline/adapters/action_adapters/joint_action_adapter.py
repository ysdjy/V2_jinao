"""Convert a canonical Action into a joint-space IsaacLab action vector.

Target env family:
  * Isaac-*-Franka-Joint*    -> action = [q1..q7, gripper]  (8D, abs joint position)
    (the user's existing joint state-machine envs use this layout)

If the policy emits delta_ee_pose but the env needs joints, you need an IK step
on the IsaacLab side (the env provides the Jacobian). This adapter only handles
the case where the policy already emits joint_position / joint_delta actions.
"""

from __future__ import annotations

from typing import Any, Optional


def build_joint_action(
    action: dict[str, Any],
    current_joints: Optional[list[float]] = None,
    num_joints: int = 7,
) -> list[float]:
    at = action.get("action_type", "joint_position")
    jt = action.get("joint_targets")
    grip = float(action.get("gripper", 0.0))

    if jt is None:
        raise ValueError("joint adapter requires action.joint_targets")
    jt = [float(v) for v in jt][:num_joints]

    if at == "joint_delta":
        if current_joints is None:
            raise ValueError("joint_delta needs current_joints")
        jt = [float(current_joints[i]) + jt[i] for i in range(min(num_joints, len(jt)))]

    while len(jt) < num_joints:
        jt.append(0.0)
    return jt + [grip]
