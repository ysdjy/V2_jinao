# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Calibrate the front-face handle offset for top_drawer (link_0) and middle_drawer (link_2).

The drawers open toward world -X, so the graspable front panel/handle is at the drawer mesh's
-X-most face. This tool computes, at the closed pose, the mesh AABB of each drawer link, takes the
front-face center (min world-X, centered in y/z), and converts it to a CONSTANT offset in the link's
local frame (geometry-relative, independent of where the cabinet is placed). That offset is what we
store in custom_drawer_config.py and apply identically in training (FrameTransformer / custom_drawer_mdp)
and deployment (SelectedDrawerObsAdapter). Run with --disable_fabric so USD bounds match physics.

    ./isaaclab.sh -p scripts/environments/state_machine/debug_drawer_handle_calib.py \
        --num_envs 1 --disable_fabric --seed 1 [--headless]
"""

from __future__ import annotations

import argparse

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Calibrate drawer front-face handle offsets.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--show_affordance_debug", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

import torch

import isaaclab.utils.math as math_utils
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import gymnasium as gym

from runtime.base_skill import pose_tensor
from runtime.debug_visualizer import DebugVisualizer
from runtime.scene_state_provider import PoseState, SceneStateProvider
from runtime.simple_scene_layout import SimpleSceneLayoutManager

TASK_ID = "Isaac-Stack-Cube-Franka-JointPolicy-v0"
# drawers we calibrate (bottom is locked / excluded)
CALIB = [("top_drawer", "link_0"), ("middle_drawer", "link_2"), ("bottom_drawer", "link_1")]


def _round(seq, n=4):
    return [round(float(v), n) for v in seq]


def _mesh_world_aabb(stage, prim_path):
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=True)
    bound = cache.ComputeWorldBound(prim)
    rng = bound.ComputeAlignedRange()
    mn, mx = rng.GetMin(), rng.GetMax()
    return [float(mn[0]), float(mn[1]), float(mn[2])], [float(mx[0]), float(mx[1]), float(mx[2])]


def main():
    env_cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    env_cfg.seed = args_cli.seed
    env = gym.make(TASK_ID, cfg=env_cfg)
    env.reset(seed=args_cli.seed)

    provider = SceneStateProvider(env)
    layout_manager = SimpleSceneLayoutManager(env=env, base_seed=args_cli.seed)
    visualizer = DebugVisualizer(enabled=not args_cli.headless or args_cli.show_affordance_debug)
    stage = provider.scene.stage
    cabinet = provider.scene["cabinet"]

    provider.set_sim_time(0.0)
    layout_manager.reset_layout(reset_index=0)
    for jn in ("joint_0", "joint_1", "joint_2"):
        provider.reset_cabinet_joint(jn, 0.0)
    state = provider.get_state()
    for _ in range(5):
        env.step(provider.make_hold_joint_action(state, 1.0))

    body_names = list(cabinet.data.body_names)
    print("\n========== HANDLE CALIBRATION (front face = min world-X) ==========", flush=True)
    print("Paste these offsets into custom_drawer_config.py HANDLE_LOCAL_OFFSET:\n", flush=True)
    for target, link in CALIB:
        link_idx = body_names.index(link)
        link_pos = cabinet.data.body_pos_w[provider.env_id, link_idx]
        link_quat = cabinet.data.body_quat_w[provider.env_id, link_idx]
        aabb = _mesh_world_aabb(stage, f"/World/envs/env_0/Cabinet/{link}")
        if aabb is None:
            print(f"  {target} ({link}): could not read mesh AABB", flush=True)
            continue
        mn, mx = aabb
        # front-face center: min world-Y, centered in x/z (rotated cabinet: drawers open toward -Y)
        handle_world = torch.tensor(
            [0.5 * (mn[0] + mx[0]), mn[1], 0.5 * (mn[2] + mx[2])], dtype=torch.float32, device=link_pos.device
        )
        # convert to a constant offset in the link's local frame
        rel_world = (handle_world - link_pos).unsqueeze(0)
        offset_local = math_utils.quat_apply(math_utils.quat_inv(link_quat).unsqueeze(0), rel_world)[0]
        print(
            f"  {target} ({link}):\n"
            f"      mesh_aabb_world   min={_round(mn)} max={_round(mx)}\n"
            f"      link_pos_world    = {_round(link_pos.tolist())}\n"
            f"      handle_world      = {_round(handle_world.tolist())}  (front-face center)\n"
            f"      OFFSET_LOCAL      = {_round(offset_local.tolist())}  <-- store this\n",
            flush=True,
        )
        # markers: link frame + computed handle
        visualizer.update_pose(f"calib_{link}_frame", pose_tensor(PoseState(link_pos, link_quat)), use_coordinate_arrows=True)
        quat_id = torch.tensor([1.0, 0.0, 0.0, 0.0], device=link_pos.device)
        visualizer.update_pose(f"calib_{target}_handle", torch.cat((handle_world, quat_id)))
    print("===================================================================\n", flush=True)

    if not args_cli.headless:
        print("[calib] holding viewport; markers show link frame + handle. Close window to exit.", flush=True)
        while simulation_app.is_running():
            st = provider.get_state()
            env.step(provider.make_hold_joint_action(st, 1.0))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
