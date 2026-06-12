"""Template observation adapter for a CUSTOM IsaacSim task.

Copy this file, rename, and edit `build_custom_observation` to map YOUR env's
scene entities / sensors into the unified Observation schema (see
adapters/policy_server/schemas.py). The default rollout runner can then drive
your task unchanged.

Key things to customize:
  * robot articulation key in the scene (default "robot")
  * end-effector body name
  * camera sensor names -> logical front_rgb / wrist_rgb
  * objects: fill from your task's object poses, or later from FoundationPose
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def build_custom_observation(
    env,
    task_instruction: str,
    episode_id: int,
    step_id: int,
    *,
    robot_key: str = "robot",
    ee_body: str = "panda_hand",
    camera_map: Optional[dict] = None,  # {"front_rgb": "table_cam", "wrist_rgb": "wrist_cam"}
    object_keys: Optional[list] = None,  # scene rigid-object keys to expose as objects[]
    image_mode: str = "none",
    image_dir: Optional[str] = None,
) -> dict:
    import time

    scene = env.unwrapped.scene
    robot = scene[robot_key]
    jp = robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
    jv = robot.data.joint_vel[0].detach().cpu().numpy().astype(np.float32)

    ee_pos, ee_quat_xyzw = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]
    try:
        names = list(robot.data.body_names)
        if ee_body in names:
            bi = names.index(ee_body)
            bs = robot.data.body_state_w[0, bi]
            root = robot.data.root_state_w[0, :3]
            p = (bs[:3] - root).detach().cpu().numpy()
            q = bs[3:7].detach().cpu().numpy()  # wxyz
            ee_pos = [float(p[0]), float(p[1]), float(p[2])]
            ee_quat_xyzw = [float(q[1]), float(q[2]), float(q[3]), float(q[0])]
    except Exception:
        pass

    objects = []
    for k in object_keys or []:
        try:
            obj = scene[k]
            pos = obj.data.root_pos_w[0].detach().cpu().numpy()
            quat = obj.data.root_quat_w[0].detach().cpu().numpy()  # wxyz
            root = robot.data.root_state_w[0, :3].detach().cpu().numpy()
            objects.append({
                "name": k,
                "position": [float(pos[0] - root[0]), float(pos[1] - root[1]), float(pos[2] - root[2])],
                "quat": [float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0])],
                "confidence": 1.0,
            })
        except Exception:
            continue

    # images: reuse the generic capture helper if you want; left as none here.
    images: dict[str, Any] = {}

    return {
        "timestamp": time.time(),
        "task_instruction": task_instruction,
        "robot": {
            "joint_positions": jp.tolist(),
            "joint_velocities": jv.tolist(),
            "ee_position": ee_pos,
            "ee_quat": ee_quat_xyzw,
            "gripper_width": float(jp[-2] + jp[-1]) if len(jp) >= 2 else 0.0,
        },
        "images": images,
        "objects": objects,
        "metadata": {"env_name": "custom", "episode_id": episode_id, "step_id": step_id},
    }
