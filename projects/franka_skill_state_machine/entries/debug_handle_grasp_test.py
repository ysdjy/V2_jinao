# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Test whether the Franka can reach and grip a drawer handle, and whether the pose is sane.

Drives the TCP (via the encapsulated DLS IK) to a front-approach grasp at the selected drawer's
handle (the custom cabinet faces -Y toward the robot, so the gripper approaches moving +Y with the
fingers straddling the horizontal handle bar along world Z), then closes the gripper. Logs
reachability (position/orientation error, ik_success), the resulting arm joint configuration, and
the gripper width after closing. Run with GUI to watch the pose; --headless for numbers only.

    ./isaaclab.sh -p scripts/environments/state_machine/debug_handle_grasp_test.py \
        --target_drawer top_drawer --seed 1
"""

from __future__ import annotations

import argparse

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Drawer-handle grasp/reach pose test.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--target_drawer", type=str, default="top_drawer", choices=["top_drawer", "middle_drawer"])
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--max_steps", type=int, default=1200)
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
from runtime.ik_joint_adapter import IKJointAdapter
from runtime.scene_state_provider import SceneStateProvider
from runtime.simple_scene_layout import SimpleSceneLayoutManager

TASK_ID = "Isaac-Stack-Cube-Franka-JointPolicy-v0"


def _front_grasp_quat(device):
    # hand axes in world: approach z=+Y (into the handle face), finger-open y=+Z (straddle bar
    # top/bottom), x = y cross z = -X. R columns = [x_hand, y_hand, z_hand].
    R = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], device=device)
    return math_utils.quat_from_matrix(R.unsqueeze(0))[0]


def main():
    env_cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    env_cfg.seed = args_cli.seed
    env_cfg.viewer.eye = (2.0, -1.5, 1.4)
    env_cfg.viewer.lookat = (0.35, 0.5, 0.4)
    env = gym.make(TASK_ID, cfg=env_cfg)
    env.reset(seed=args_cli.seed)

    provider = SceneStateProvider(env)
    layout = SimpleSceneLayoutManager(env=env, base_seed=args_cli.seed)
    adapter = IKJointAdapter(env)
    obs_adapter = SelectedDrawerObsAdapter(env, args_cli.target_drawer)
    visualizer = DebugVisualizer(enabled=not args_cli.headless)

    sim_dt = env_cfg.sim.dt * env_cfg.decimation
    provider.set_sim_time(0.0)
    layout.reset_layout(reset_index=0)
    provider.reset_cabinet_joint("joint_0", 0.0)
    state = provider.get_state()
    for _ in range(5):
        env.step(provider.make_hold_joint_action(state, 1.0))

    handle = obs_adapter.selected_handle_pos_w()[0].clone()
    quat = _front_grasp_quat(env.unwrapped.device)
    pre_grasp = PoseState(handle + torch.tensor([0.0, -0.12, 0.0], device=handle.device), quat)
    grasp = PoseState(handle.clone(), quat)
    print(
        f"[handle_grasp] target_drawer={args_cli.target_drawer} handle_world="
        f"{[round(float(v),4) for v in handle.tolist()]}",
        flush=True,
    )

    phases = [("MOVE_PRE_GRASP", pre_grasp, 1.0, 0.018), ("APPROACH", grasp, 1.0, 0.015), ("CLOSE", grasp, -1.0, None)]
    step_count = 0
    last_q = None
    for phase_name, target, gripper, reach_tol in phases:
        stable = 0
        close_steps = int(1.2 / sim_dt)
        ticks = 0
        while step_count < args_cli.max_steps:
            state = provider.get_state()
            cur = state.robot.tcp_pose
            cmd_pose = step_pose(cur, target, max_pos_step=0.01, max_ori_step=math.radians(5.0))
            ik = adapter.solve(cmd_pose)
            err = pose_error(cur, target)
            if ik.success:
                last_q = ik.q_des
                action = provider.make_joint_action_from_q_des(ik.q_des, gripper)
            else:
                action = provider.make_hold_joint_action(state, gripper)
            visualizer.update_pose("handle_target", pose_tensor(target), use_coordinate_arrows=True)
            visualizer.update_pose("handle_pt", torch.cat((handle, quat)))
            env.step(action)
            step_count += 1
            ticks += 1
            if step_count % 30 == 0:
                print(
                    f"[handle_grasp] {phase_name} step={step_count} pos_err={err.position:.4f} "
                    f"ori_err_deg={math.degrees(err.orientation):.2f} ik_success={ik.success} "
                    f"gripper_w={state.robot.gripper_width:.4f} "
                    f"arm_q={[round(float(v),3) for v in state.robot.joint_pos[adapter._joint_ids].tolist()]}",
                    flush=True,
                )
            if reach_tol is not None:
                if err.position <= reach_tol:
                    stable += 1
                    if stable >= 8:
                        break
                else:
                    stable = 0
            elif ticks >= close_steps:
                break

    state = provider.get_state()
    err = pose_error(state.robot.tcp_pose, grasp)
    q = state.robot.joint_pos[adapter._joint_ids].tolist()
    within = adapter._joint_lower.tolist(), adapter._joint_upper.tolist()
    print("\n========== HANDLE GRASP RESULT ==========", flush=True)
    print(f"target_drawer={args_cli.target_drawer} handle={[round(float(v),4) for v in handle.tolist()]}", flush=True)
    print(f"final tcp pos_err={err.position:.4f} m  ori_err={math.degrees(err.orientation):.2f} deg", flush=True)
    print(f"final gripper_width={state.robot.gripper_width:.4f} (closed-on-handle if >0 and <0.08)", flush=True)
    print(f"final arm_q={[round(float(v),3) for v in q]}", flush=True)
    print(f"joint_lower={[round(v,2) for v in within[0]]}", flush=True)
    print(f"joint_upper={[round(v,2) for v in within[1]]}", flush=True)
    print(f"reached={'YES' if err.position < 0.03 else 'NO'}", flush=True)
    print("=========================================\n", flush=True)

    if not args_cli.headless:
        print("[handle_grasp] holding; close the window to exit.", flush=True)
        while simulation_app.is_running():
            state = provider.get_state()
            if last_q is not None:
                env.step(provider.make_joint_action_from_q_des(last_q, -1.0))
            else:
                env.step(provider.make_hold_joint_action(state, -1.0))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
