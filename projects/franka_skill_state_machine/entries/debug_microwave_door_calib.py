# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Calibrate the microwave door (revolute joint_0) geometry for the open/close-door skill.

The microwave is a fixed-base articulation added in stack_joint_pos_env_cfg.py (prim /Microwave).
joint_0 is the revolute door hinge: body0=link_1 (cabinet body), body1=link_0 (the DOOR), local
axis = Y, limits [0deg, 90deg]. Because the joint anchor on body1 is at link_0's origin, link_0's
body origin IS the hinge point and the world hinge axis = R(link_0_quat) @ local_Y.

This tool, at the deployed microwave pose:
  1. reads link_0 (door) body world pose -> hinge position + hinge axis,
  2. sweeps joint_0 through several angles and reads the door mesh world AABB + door body pose so we
     can see the swing and confirm the hinge stays fixed,
  3. computes a graspable HANDLE point (door front face, far edge from the hinge, mid height) and
     converts it to a CONSTANT offset in link_0's local frame (geometry-relative). That offset is
     what the handle collision proxy and the MicrowaveDoorObsAdapter use.

Run with --disable_fabric so USD mesh bounds match physics.

    ./isaaclab.sh -p projects/franka_skill_state_machine/entries/debug_microwave_door_calib.py \
        --num_envs 1 --disable_fabric --seed 1 --headless
"""

from __future__ import annotations

import argparse

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Calibrate microwave door geometry.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--angles", type=str, default="0,30,60,90", help="joint_0 sweep angles in degrees.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

import math

import torch

import isaaclab.utils.math as math_utils
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import gymnasium as gym

from runtime.scene_state_provider import SceneStateProvider
from runtime.simple_scene_layout import SimpleSceneLayoutManager

TASK_ID = "Isaac-Stack-Cube-Franka-JointPolicy-v0"
DOOR_LINK = "link_0"
HINGE_JOINT = "joint_0"


def _round(seq, n=4):
    return [round(float(v), n) for v in seq]


def _mesh_world_aabb(stage, prim_path):
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=True
    )
    bound = cache.ComputeWorldBound(prim)
    rng = bound.ComputeAlignedRange()
    mn, mx = rng.GetMin(), rng.GetMax()
    return [float(mn[0]), float(mn[1]), float(mn[2])], [float(mx[0]), float(mx[1]), float(mx[2])]


def main():
    env_cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    env_cfg.seed = args_cli.seed
    # Free the microwave door joint (like the cabinet drawer for ik_pull) so write_joint_state holds
    # cleanly and a gripper could physically pull it. The hinge axis is vertical, so gravity exerts
    # no torque about it -> a freed door stays where placed.
    if hasattr(env_cfg.scene, "microwave") and hasattr(env_cfg.scene.microwave, "actuators"):
        for act in env_cfg.scene.microwave.actuators.values():
            act.stiffness = 0.0
            act.damping = 2.0
    env = gym.make(TASK_ID, cfg=env_cfg)
    env.reset(seed=args_cli.seed)

    provider = SceneStateProvider(env)
    layout_manager = SimpleSceneLayoutManager(env=env, base_seed=args_cli.seed)
    stage = provider.scene.stage
    mw = provider.scene["microwave"]
    eid = provider.env_id

    provider.set_sim_time(0.0)
    layout_manager.reset_layout(reset_index=0)
    state = provider.get_state()
    for _ in range(5):
        env.step(provider.make_hold_joint_action(state, 1.0))

    jnames = list(mw.data.joint_names)
    bnames = list(mw.data.body_names)
    print("\n========== MICROWAVE DOOR CALIBRATION ==========", flush=True)
    print(f"joint_names = {jnames}", flush=True)
    print(f"body_names  = {bnames}", flush=True)
    jid = jnames.index(HINGE_JOINT)
    door_idx = bnames.index(DOOR_LINK)

    angles = [float(a) for a in args_cli.angles.split(",") if a.strip() != ""]

    def door_pose():
        return mw.data.body_pos_w[eid, door_idx].clone(), mw.data.body_quat_w[eid, door_idx].clone()

    def hold_door(angle_deg: float):
        # door joint is FREE (stiffness 0); just teleport it and let it settle (no position target)
        ang = math.radians(angle_deg)
        q = mw.data.joint_pos.clone()
        q[:, jid] = ang
        mw.write_joint_state_to_sim(q, torch.zeros_like(q))
        st = provider.get_state()
        for _ in range(15):
            env.step(provider.make_hold_joint_action(st, 1.0))

    dev = mw.data.body_pos_w.device

    # --- closed-pose geometry (reliable) ---------------------------------------
    hold_door(0.0)
    closed_door_pos, closed_door_quat = door_pose()
    closed_aabb = _mesh_world_aabb(stage, f"/World/envs/env_0/Microwave/{DOOR_LINK}")

    local_y = torch.tensor([0.0, 1.0, 0.0], device=dev)
    hinge_axis_w = math_utils.quat_apply(closed_door_quat.unsqueeze(0), local_y.unsqueeze(0))[0]
    hinge_axis_w = hinge_axis_w / torch.linalg.norm(hinge_axis_w)

    # handle = door free edge (AABB corner farthest from the hinge, horizontally), mid height
    mn, mx = closed_aabb
    hinge_xy = closed_door_pos[:2]
    midz = 0.5 * (mn[2] + mx[2])
    corners = [
        torch.tensor([mn[0], mn[1], midz], device=dev),
        torch.tensor([mn[0], mx[1], midz], device=dev),
        torch.tensor([mx[0], mn[1], midz], device=dev),
        torch.tensor([mx[0], mx[1], midz], device=dev),
    ]
    handle_world = max(corners, key=lambda c: float(torch.linalg.norm(c[:2] - hinge_xy)))
    rel_world = (handle_world - closed_door_pos).unsqueeze(0)
    handle_offset_local = math_utils.quat_apply(
        math_utils.quat_inv(closed_door_quat).unsqueeze(0), rel_world
    )[0]

    def live_handle():
        dp, dq = door_pose()
        return math_utils.quat_apply(dq.unsqueeze(0), handle_offset_local.unsqueeze(0))[0] + dp

    def analytic_handle(theta_rad: float):
        # rotate (handle_closed - hinge) about world Z by theta, add hinge
        c, s = math.cos(theta_rad), math.sin(theta_rad)
        rel = handle_world - closed_door_pos
        rx = c * float(rel[0]) - s * float(rel[1])
        ry = s * float(rel[0]) + c * float(rel[1])
        return torch.tensor([float(closed_door_pos[0]) + rx, float(closed_door_pos[1]) + ry, float(handle_world[2])], device=dev)

    # --- sweep: confirm the door physically holds each angle and the handle follows the arc -------
    print("\n-- joint_0 sweep (door=link_0, joint FREE) : live handle vs analytic Z-arc --", flush=True)
    for ang in angles:
        hold_door(ang)
        real = math.degrees(float(mw.data.joint_pos[eid, jid]))
        lh = live_handle()
        ah_pos = analytic_handle(math.radians(ang))
        ah_neg = analytic_handle(math.radians(-ang))
        err_pos = float(torch.linalg.norm(lh - ah_pos))
        err_neg = float(torch.linalg.norm(lh - ah_neg))
        sign = "+" if err_pos <= err_neg else "-"
        print(
            f"  cmd={ang:6.1f}deg real={real:7.2f}deg live_handle={_round(lh.tolist())} "
            f"arc_err(+)={err_pos:.4f} arc_err(-)={err_neg:.4f} best_sign={sign}",
            flush=True,
        )
    hold_door(0.0)

    print("\n-- DERIVED (closed pose) --", flush=True)
    print(f"  hinge_pos_world   = {_round(closed_door_pos.tolist())}  (= link_0 origin)", flush=True)
    print(f"  hinge_axis_world  = {_round(hinge_axis_w.tolist())}", flush=True)
    print(f"  door_quat_world   = {_round(closed_door_quat.tolist())}", flush=True)
    print(f"  handle_world      = {_round(handle_world.tolist())}  (far free edge, mid height)", flush=True)
    print(f"  HANDLE_OFFSET_LOCAL (link_0 frame) = {_round(handle_offset_local.tolist())}  <-- store this", flush=True)
    print(f"  handle radius (|handle-hinge| horiz) = {float(torch.linalg.norm((handle_world[:2]-hinge_xy))):.4f} m", flush=True)
    print("================================================\n", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
