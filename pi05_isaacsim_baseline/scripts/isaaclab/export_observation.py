"""Start an IsaacLab Franka task, read one observation, save it as a sample.

Run with the IsaacLab launcher:
  ./isaaclab.sh -p pi05_isaacsim_baseline/scripts/isaaclab/export_observation.py \
      --task Isaac-Stack-Cube-Franka-IK-Rel-v0 --headless

Saves data/processed/sample_observation.json (image_mode=none by default).
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Export a sample observation from an IsaacLab task.")
parser.add_argument("--task", type=str, default="Isaac-Stack-Cube-Franka-IK-Rel-v0")
parser.add_argument("--image_mode", choices=["none", "path", "base64"], default="none")
parser.add_argument("--task_instruction", type=str, default="Stack the cubes with the Franka robot.")
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--steps", type=int, default=3, help="warm-up steps before capturing")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import os
import sys

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
import isaac_obs_utils as obs_utils  # noqa: E402


def main():
    out = args_cli.out or os.path.join(_PROJ, "data", "processed", "sample_observation.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    img_dir = os.path.join(_PROJ, "data", "processed", "images")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    try:
        env_cfg.observations.policy.concatenate_terms = False
    except Exception:
        pass
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    # warm up a few steps with zero action so state is meaningful
    a_dim = env.action_space.shape[-1]
    zero = torch.zeros((env.unwrapped.num_envs, a_dim), device=env.unwrapped.device)
    for _ in range(max(0, args_cli.steps)):
        env.step(zero)

    obs = obs_utils.build_observation(
        env, args_cli.task_instruction, 0, args_cli.steps,
        image_mode=args_cli.image_mode, image_dir=img_dir,
    )
    obs["_meta_action_dim"] = int(a_dim)
    obs["_meta_env_kind"] = obs_utils.detect_env_kind(args_cli.task)
    with open(out, "w") as f:
        json.dump(obs, f, indent=2)
    print(f"[export_observation] saved sample observation -> {out}")
    print(f"[export_observation] action_dim={a_dim} env_kind={obs['_meta_env_kind']}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
