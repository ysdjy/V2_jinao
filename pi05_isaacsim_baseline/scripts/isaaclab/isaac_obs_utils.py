"""Helpers to build a unified Observation from an IsaacLab manager-based env and
to apply a canonical Action back onto the env. Imported by the IsaacLab-side
scripts (run with ./isaaclab.sh -p).

These read robot state directly from the scene articulation so they do not depend
on a particular observation-term layout. End-effector pose is taken from the hand
body if present, else from an obs term, else zeros.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np


# --------------------------------------------------------------------------- #
def detect_env_kind(task_name: str) -> str:
    """Return one of {'ik_rel','ik_abs','joint'} from the task name."""
    t = task_name.lower()
    if "ik-abs" in t or "ik_abs" in t:
        return "ik_abs"
    if "ik-rel" in t or "ik_rel" in t:
        return "ik_rel"
    if "joint" in t:
        return "joint"
    # default for stack/lift franka teleop tasks is relative IK
    return "ik_rel"


def _find_hand_body_index(robot) -> Optional[int]:
    names = list(getattr(robot.data, "body_names", []) or getattr(robot, "body_names", []))
    for cand in ["panda_hand", "panda_link8", "ee_link", "tool0", "gripper"]:
        if cand in names:
            return names.index(cand)
    return None


def build_observation(
    env,
    task_instruction: str,
    episode_id: int,
    step_id: int,
    image_mode: str = "none",
    image_dir: Optional[str] = None,
    cam_keys=("front_rgb", "wrist_rgb"),
) -> dict:
    """Return Observation.to_dict()-compatible dict (plain python types)."""
    import time

    scene = env.unwrapped.scene
    robot = scene["robot"]
    jp = robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
    jv = robot.data.joint_vel[0].detach().cpu().numpy().astype(np.float32)

    # End-effector pose
    ee_pos = [0.0, 0.0, 0.0]
    ee_quat_xyzw = [0.0, 0.0, 0.0, 1.0]
    hand_idx = _find_hand_body_index(robot)
    try:
        if hand_idx is not None:
            body_state = robot.data.body_state_w[0, hand_idx]  # [px,py,pz, qw,qx,qy,qz, ...]
            root = robot.data.root_state_w[0, :3]
            p = (body_state[:3] - root).detach().cpu().numpy()
            qwxyz = body_state[3:7].detach().cpu().numpy()
            ee_pos = [float(p[0]), float(p[1]), float(p[2])]
            ee_quat_xyzw = [float(qwxyz[1]), float(qwxyz[2]), float(qwxyz[3]), float(qwxyz[0])]
    except Exception:
        pass

    # gripper width: last two finger joints if present
    gripper_width = 0.0
    try:
        gripper_width = float(jp[-2] + jp[-1])
    except Exception:
        pass

    images: dict[str, Any] = {}
    if image_mode != "none":
        images = _capture_images(scene, image_mode, image_dir, episode_id, step_id, cam_keys)

    return {
        "timestamp": time.time(),
        "task_instruction": task_instruction,
        "robot": {
            "joint_positions": jp.tolist(),
            "joint_velocities": jv.tolist(),
            "ee_position": ee_pos,
            "ee_quat": ee_quat_xyzw,
            "gripper_width": gripper_width,
        },
        "images": images,
        "objects": [],
        "metadata": {"env_name": getattr(env.unwrapped, "cfg", None).__class__.__name__ if hasattr(env.unwrapped, "cfg") else "",
                     "episode_id": episode_id, "step_id": step_id},
    }


def _capture_images(scene, image_mode, image_dir, episode_id, step_id, cam_keys):
    out: dict[str, Any] = {}
    cam_name_map = {"front_rgb": ["table_cam", "front_cam", "camera"], "wrist_rgb": ["wrist_cam", "hand_cam"]}
    for logical, candidates in cam_name_map.items():
        if logical not in cam_keys:
            continue
        sensor = None
        for c in candidates:
            try:
                sensor = scene[c]
                break
            except Exception:
                continue
        if sensor is None:
            continue
        try:
            rgb = sensor.data.output["rgb"][0].detach().cpu().numpy()
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            rgb = rgb[..., :3]
            if image_mode == "path" and image_dir:
                ep_dir = os.path.join(image_dir, f"ep{episode_id:04d}")
                os.makedirs(ep_dir, exist_ok=True)
                from PIL import Image

                fp = os.path.join(ep_dir, f"{logical}_step{step_id:06d}.png")
                Image.fromarray(rgb).save(fp)
                out[logical] = {"mode": "path", "path": fp, "shape": list(rgb.shape)}
            elif image_mode == "base64":
                import base64
                import io

                from PIL import Image

                buf = io.BytesIO()
                Image.fromarray(rgb).save(buf, format="JPEG")
                out[logical] = {"mode": "base64", "base64": base64.b64encode(buf.getvalue()).decode(),
                                "shape": list(rgb.shape)}
        except Exception:
            continue
    return out


# --------------------------------------------------------------------------- #
def action_to_env_tensor(action: dict, env_kind: str, obs_dict: dict, num_envs: int, device):
    """Convert a canonical Action dict into a torch action tensor (num_envs, A)."""
    import torch

    robot = obs_dict.get("robot", {})
    if env_kind == "ik_rel":
        dpos = list(action.get("delta_ee_position", [0, 0, 0]))[:3]
        drot = list(action.get("delta_ee_rot", [0, 0, 0]))[:3]
        grip = float(action.get("gripper", 0.0))
        vec = dpos + drot + [grip]
    elif env_kind == "ik_abs":
        import math

        ee = robot.get("ee_position", [0, 0, 0])
        q = robot.get("ee_quat", [0, 0, 0, 1])  # xyzw
        dpos = list(action.get("delta_ee_position", [0, 0, 0]))[:3]
        drot = list(action.get("delta_ee_rot", [0, 0, 0]))[:3]
        grip = float(action.get("gripper", 0.0))
        new_pos = [ee[i] + dpos[i] for i in range(3)]
        ang = math.sqrt(sum(c * c for c in drot))
        if ang < 1e-8:
            dq = [1.0, 0, 0, 0]
        else:
            ax = [c / ang for c in drot]
            s = math.sin(ang / 2)
            dq = [math.cos(ang / 2), ax[0] * s, ax[1] * s, ax[2] * s]
        cw = [q[3], q[0], q[1], q[2]]
        nw = [
            dq[0] * cw[0] - dq[1] * cw[1] - dq[2] * cw[2] - dq[3] * cw[3],
            dq[0] * cw[1] + dq[1] * cw[0] + dq[2] * cw[3] - dq[3] * cw[2],
            dq[0] * cw[2] - dq[1] * cw[3] + dq[2] * cw[0] + dq[3] * cw[1],
            dq[0] * cw[3] + dq[1] * cw[2] - dq[2] * cw[1] + dq[3] * cw[0],
        ]
        n = math.sqrt(sum(c * c for c in nw)) or 1.0
        nw = [c / n for c in nw]
        vec = new_pos + nw + [grip]
    elif env_kind == "joint":
        jt = action.get("joint_targets")
        grip = float(action.get("gripper", 0.0))
        if jt is None:
            jt = list(robot.get("joint_positions", [0.0] * 7))[:7]
        vec = list(jt)[:7] + [grip]
    else:
        raise ValueError(f"unknown env_kind {env_kind}")

    t = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
    return t.repeat(num_envs, 1)
