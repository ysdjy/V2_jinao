# Copyright (c) 2026. KitchenV0 project.
# SPDX-License-Identifier: BSD-3-Clause

"""Run a KitchenV0 environment with zero actions.

Example:
    ./isaaclab.sh -p KitchenV0/scripts/zero_agent.py --task Kitchen-V0-Franka-IK-Abs-Play-v0 --num_envs 1
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Zero-action runner for KitchenV0 environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Kitchen-V0-Franka-IK-Abs-Play-v0", help="Name of the task.")
parser.add_argument("--max_steps", type=int, default=0, help="If > 0, stop after this many environment steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import kitchen_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    """Run the environment with zero actions."""

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    env.reset()

    step_count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)
        step_count += 1
        if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
            break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
