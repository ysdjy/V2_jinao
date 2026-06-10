# Copyright (c) 2026. Connection project.
# SPDX-License-Identifier: BSD-3-Clause

"""Load a Connection task scene, hold the arm still, and record a short video so the
layout (asset placement / reachability) can be inspected without running a policy."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Static scene viewer / layout check.")
parser.add_argument("--task", type=str, default="Connection-Multi-Skill-Franka-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=120)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import gymnasium as gym
import torch

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import connection_tasks  # noqa: F401
from connection_tasks.tasks.multi_skill.multi_skill_env_cfg import (
    CABINET_POS_LOCAL,
    CABINET_SCENE_ORIGIN,
    KNIFE_POS_LOCAL,
)


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    video_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "videos", "scene_view"))
    os.makedirs(video_dir, exist_ok=True)
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_dir,
        step_trigger=lambda step: step == 0,
        video_length=args_cli.steps,
        disable_logger=True,
    )
    print(f"[view_scene] recording to {video_dir}", flush=True)

    env.reset()
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    idx = 0
    for name, dim in zip(env.unwrapped.action_manager.active_terms, env.unwrapped.action_manager.action_term_dim):
        if "arm_action" in name and dim == 7 and "IK-Abs" in args_cli.task:
            actions[:, idx + 3] = 1.0
        if "gripper_action" in name:
            actions[:, idx] = 1.0
        idx += dim
    for i in range(args_cli.steps):
        if not simulation_app.is_running():
            break
        with torch.inference_mode():
            env.step(actions)
        if i == args_cli.steps - 2:
            _print_diagnostics(env)
    env.close()


def _print_diagnostics(env):
    scene = env.unwrapped.scene
    env_origin = scene.env_origins[0]

    def station_origin(local_origin):
        return env_origin + torch.tensor(local_origin, device=env_origin.device)

    def rel(p, local_origin):
        origin = station_origin(local_origin)
        return tuple(round(v, 3) for v in (p[0] - origin).detach().to("cpu").tolist())

    def dist(p, local_origin):
        origin = station_origin(local_origin)
        return round(float((p[0] - origin).norm().detach().cpu()), 3)

    cabinet_robot = scene["cabinet_robot"].data.root_pos_w
    cabinet_root = scene["cabinet"].data.root_pos_w
    cabinet_ee = scene["cabinet_ee_frame"].data.target_pos_w[:, 0, :]
    cab_h = scene["cabinet_frame"].data.target_pos_w[:, 0, :]
    knife = scene["knife"].data.root_pos_w
    opened_handle_est = cab_h + torch.tensor((-0.24, 0.0, 0.0), device=cab_h.device)
    print("=== CABINET LAYOUT DIAGNOSTICS (station frame) ===", flush=True)
    print(f"  configured cabinet_pos={CABINET_POS_LOCAL} knife_pos={KNIFE_POS_LOCAL}", flush=True)
    print(
        f"  robot_root       pos={rel(cabinet_robot, CABINET_SCENE_ORIGIN)}    "
        f"dist={dist(cabinet_robot, CABINET_SCENE_ORIGIN)}",
        flush=True,
    )
    print(
        f"  cabinet_root     pos={rel(cabinet_root, CABINET_SCENE_ORIGIN)}    "
        f"dist={dist(cabinet_root, CABINET_SCENE_ORIGIN)}",
        flush=True,
    )
    print(
        f"  cabinet_tcp      pos={rel(cabinet_ee, CABINET_SCENE_ORIGIN)}    "
        f"dist={dist(cabinet_ee, CABINET_SCENE_ORIGIN)}",
        flush=True,
    )
    print(
        f"  bottom_drawer_h  pos={rel(cab_h, CABINET_SCENE_ORIGIN)}  "
        f"dist={dist(cab_h, CABINET_SCENE_ORIGIN)}",
        flush=True,
    )
    print(
        f"  opened_handle*   pos={rel(opened_handle_est, CABINET_SCENE_ORIGIN)}  "
        f"dist={dist(opened_handle_est, CABINET_SCENE_ORIGIN)}",
        flush=True,
    )
    print(
        f"  knife_root       pos={rel(knife, CABINET_SCENE_ORIGIN)}  "
        f"dist={dist(knife, CABINET_SCENE_ORIGIN)}",
        flush=True,
    )
    print("  *opened_handle is estimated at bottom drawer joint ~= 0.24 m", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
