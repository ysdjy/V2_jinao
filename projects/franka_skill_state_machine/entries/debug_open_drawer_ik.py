# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Open a drawer PHYSICALLY via IK (no learned policy, no joint-target cheating).

Sequence: reach the handle (front approach, +Y) -> close the gripper on the handle bar -> pull the
TCP in the drawer's opening direction (world -Y) while gripping. The drawer follows because the
gripper holds the handle. We NEVER call set_cabinet_joint_target; the drawer joint moves only from
physical interaction. Success = drawer joint position >= 0.20.

The drawer actuator is set to free-sliding (stiffness 0) so the pull behaves like a real drawer.

    ./isaaclab.sh -p scripts/environments/state_machine/debug_open_drawer_ik.py \
        --target_drawer top_drawer --seed 1            # GUI
        ... --headless                                  # numbers only
"""

from __future__ import annotations

import argparse

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Physically open a drawer with IK (no policy).")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--target_drawer", type=str, default="top_drawer", choices=["top_drawer", "middle_drawer"])
parser.add_argument("--open_distance", type=float, default=0.25, help="How far to pull (m, world -Y).")
parser.add_argument("--success_threshold", type=float, default=0.20)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--max_steps", type=int, default=2000)
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

from runtime.base_skill import PoseState, pose_error, pose_tensor, step_pose
from runtime.debug_visualizer import DebugVisualizer
from runtime.drawer_obs_adapter import SelectedDrawerObsAdapter
from runtime.drawer_target_config import DRAWER_TARGETS
from runtime.ik_joint_adapter import IKJointAdapter
from runtime.scene_state_provider import SceneStateProvider
from runtime.simple_scene_layout import SimpleSceneLayoutManager

TASK_ID = "Isaac-Stack-Cube-Franka-JointPolicy-v0"


def _front_grasp_quat(device):
    R = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], device=device)
    return math_utils.quat_from_matrix(R.unsqueeze(0))[0]


def main():
    env_cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    env_cfg.seed = args_cli.seed
    # free-sliding drawer so a physical pull can open it (no spring fighting the gripper)
    if hasattr(env_cfg.scene, "cabinet") and "drawers" in env_cfg.scene.cabinet.actuators:
        env_cfg.scene.cabinet.actuators["drawers"].stiffness = 0.0
        env_cfg.scene.cabinet.actuators["drawers"].damping = 2.0
    env_cfg.viewer.eye = (2.0, -1.5, 1.4)
    env_cfg.viewer.lookat = (0.3, 0.4, 0.4)
    env = gym.make(TASK_ID, cfg=env_cfg)
    env.reset(seed=args_cli.seed)

    provider = SceneStateProvider(env)
    layout = SimpleSceneLayoutManager(env=env, base_seed=args_cli.seed)
    adapter = IKJointAdapter(env)
    obs_adapter = SelectedDrawerObsAdapter(env, args_cli.target_drawer)
    visualizer = DebugVisualizer(enabled=not args_cli.headless)
    joint_name = DRAWER_TARGETS[args_cli.target_drawer]["joint_name"]

    sim_dt = env_cfg.sim.dt * env_cfg.decimation
    provider.set_sim_time(0.0)
    layout.reset_layout(reset_index=0)
    provider.reset_cabinet_joint(joint_name, 0.0)
    state = provider.get_state()
    for _ in range(5):
        env.step(provider.make_hold_joint_action(state, 1.0))

    handle = obs_adapter.selected_handle_pos_w()[0].clone()
    quat = _front_grasp_quat(env.unwrapped.device)
    pre = PoseState(handle + torch.tensor([0.0, -0.12, 0.0], device=handle.device), quat)
    grasp = PoseState(handle.clone(), quat)
    pull = PoseState(handle + torch.tensor([0.0, -args_cli.open_distance, 0.0], device=handle.device), quat)

    def drawer_pos():
        return obs_adapter.selected_drawer_joint_pos()

    print(f"[open_ik] target={args_cli.target_drawer} joint={joint_name} handle={[round(float(v),4) for v in handle.tolist()]}", flush=True)

    # (name, target, gripper, reach_tol, hold_ticks)
    phases = [
        ("MOVE_PRE_GRASP", pre, 1.0, 0.018, 0),
        ("APPROACH", grasp, 1.0, 0.012, 0),
        ("CLOSE_GRIPPER", grasp, -1.0, None, int(1.0 / sim_dt)),
        ("PULL_OPEN", pull, -1.0, 0.02, 0),
    ]
    step_count = 0
    last_q = None
    success = False
    for name, target, gripper, reach_tol, hold_ticks in phases:
        stable = ticks = 0
        while step_count < args_cli.max_steps:
            state = provider.get_state()
            cur = state.robot.tcp_pose
            cmd = step_pose(cur, target, max_pos_step=0.008, max_ori_step=math.radians(5.0))
            ik = adapter.solve(cmd)
            if ik.success:
                last_q = ik.q_des
                action = provider.make_joint_action_from_q_des(ik.q_des, gripper)
            else:
                action = provider.make_hold_joint_action(state, gripper)
            visualizer.update_pose("open_target", pose_tensor(target), use_coordinate_arrows=True)
            env.step(action)
            step_count += 1
            ticks += 1
            dp = drawer_pos()
            if dp >= args_cli.success_threshold:
                success = True
            err = pose_error(cur, target)
            if step_count % 30 == 0:
                print(
                    f"[open_ik] {name} step={step_count} tcp_pos_err={err.position:.4f} "
                    f"gripper_w={state.robot.gripper_width:.4f} drawer_pos={dp:.4f} ik={ik.success}",
                    flush=True,
                )
            if success and name == "PULL_OPEN":
                break
            if reach_tol is not None:
                if err.position <= reach_tol:
                    stable += 1
                    if stable >= 8:
                        break
                else:
                    stable = 0
            elif ticks >= hold_ticks:
                break
        if success and name == "PULL_OPEN":
            break

    dp = drawer_pos()
    print("\n========== OPEN DRAWER (IK, no policy) RESULT ==========", flush=True)
    print(f"target={args_cli.target_drawer} joint={joint_name}", flush=True)
    print(f"final drawer_pos={dp:.4f}  threshold={args_cli.success_threshold:.2f}  SUCCESS={'YES' if dp>=args_cli.success_threshold else 'NO'}", flush=True)
    print(f"final gripper_width={provider.get_state().robot.gripper_width:.4f}", flush=True)
    print("set_cabinet_joint_target NEVER called (physical pull only)", flush=True)
    print("========================================================\n", flush=True)

    if not args_cli.headless:
        print("[open_ik] holding; close the window to exit.", flush=True)
        while simulation_app.is_running():
            state = provider.get_state()
            env.step(provider.make_joint_action_from_q_des(last_q, -1.0) if last_q is not None else provider.make_hold_joint_action(state, -1.0))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
