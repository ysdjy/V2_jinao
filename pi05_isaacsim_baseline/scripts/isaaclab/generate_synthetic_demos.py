# Copyright (c) 2026. Project: pi05_isaacsim_baseline (teleop module).
# SPDX-License-Identifier: BSD-3-Clause
"""Generate SYNTHETIC (throwaway) demonstrations for the Isaac-Stack-Cube-Franka-IK-Rel task.

PURPOSE
-------
These demos are NOT task-successful and are NOT for real training. They exist only so the
*data + training plumbing* (HDF5 -> LeRobot -> pi0.5 train smoke test) can be validated
**fully autonomously**, with no human at the keyboard. Real, high-quality demos come later
from a human at the keyboard (`scripts/collect_demos.sh`) or the GELLO leader arm.

It reuses IsaacLab's own ``ActionStateRecorderManagerCfg`` so the produced HDF5 has the
*exact same structure* as ``record_demos.py`` output (same obs/action fields, same
``data/demo_*`` layout) — only the action source differs (scripted/random instead of teleop)
and the export mode is EXPORT_ALL (so episodes are kept regardless of task success).

This is headless and deterministic; an AI agent can run it without any GUI interaction.

USAGE (via wrapper, recommended):
    bash pi05_isaacsim_baseline/scripts/gen_synthetic_demos.sh --num_demos 3 --headless

USAGE (direct):
    ./isaaclab.sh -p pi05_isaacsim_baseline/scripts/isaaclab/generate_synthetic_demos.py \
        --task Isaac-Stack-Cube-Franka-IK-Rel-v0 --num_demos 3 --episode_len 120 \
        --dataset_file pi05_isaacsim_baseline/data/raw_hdf5/synthetic_plumbing.hdf5 --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate synthetic (throwaway) demos for plumbing validation.")
parser.add_argument("--task", type=str, default="Isaac-Stack-Cube-Franka-IK-Rel-v0", help="Task name.")
parser.add_argument("--num_demos", type=int, default=3, help="Number of synthetic episodes to export.")
parser.add_argument("--episode_len", type=int, default=120, help="Steps per synthetic episode.")
parser.add_argument(
    "--dataset_file",
    type=str,
    default="pi05_isaacsim_baseline/data/raw_hdf5/synthetic_plumbing.hdf5",
    help="Output HDF5 path (will be created).",
)
parser.add_argument(
    "--action_mode",
    type=str,
    default="scripted",
    choices=["scripted", "random"],
    help="scripted = smooth reach/grasp/lift pattern; random = small bounded random deltas.",
)
parser.add_argument("--seed", type=int, default=0, help="Determinism seed.")
parser.add_argument("--max_delta", type=float, default=0.02, help="Max |position delta| per step (m). Kept small.")
parser.add_argument("--max_rot_delta", type=float, default=0.03, help="Max |rotation delta| per step (rad).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- everything below runs after the app is up ----
import os

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def _resolve_dataset_path(path: str) -> str:
    """Resolve relative dataset paths against the IsaacLab repo root (two levels above this file)."""
    if os.path.isabs(path):
        return path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    return os.path.join(repo_root, path)


def _scripted_action(t: int, episode_len: int, device: torch.device, max_d: float, max_r: float) -> torch.Tensor:
    """A smooth, bounded reach -> close -> lift -> nudge pattern. NOT a real task solution.

    Returns a (1, 7) tensor: [dx, dy, dz, drx, dry, drz, gripper].
    """
    a = torch.zeros((1, 7), device=device)
    p = t / max(1, episode_len)
    if p < 0.30:  # descend, gripper open
        a[0, 2] = -max_d
        a[0, 6] = 1.0
    elif p < 0.45:  # close gripper, hold
        a[0, 6] = -1.0
    elif p < 0.75:  # lift, gripper closed
        a[0, 2] = max_d
        a[0, 6] = -1.0
    else:  # gentle lateral nudge, gripper closed
        a[0, 0] = 0.5 * max_d
        a[0, 5] = 0.5 * max_r
        a[0, 6] = -1.0
    return a


def main() -> None:
    torch.manual_seed(args_cli.seed)

    dataset_file = _resolve_dataset_path(args_cli.dataset_file)
    output_dir = os.path.dirname(dataset_file)
    output_file_name = os.path.splitext(os.path.basename(dataset_file))[0]
    os.makedirs(output_dir, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.env_name = args_cli.task.split(":")[-1]

    # Run episodes of a controlled fixed length: disable success + time_out auto-termination.
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    # Same recorder IsaacLab uses for real demos -> identical HDF5 structure. EXPORT_ALL keeps every episode.
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    device = env.device

    action_dim = int(env.action_space.shape[-1])
    print(f"[gen] task={args_cli.task} action_dim={action_dim} num_demos={args_cli.num_demos} "
          f"episode_len={args_cli.episode_len} mode={args_cli.action_mode}", flush=True)
    print("[gen] NOTE: these are SYNTHETIC throwaway demos for plumbing validation only.", flush=True)

    env.reset(seed=args_cli.seed)  # initial reset (empty episode; nothing exported)

    exported = 0
    with torch.inference_mode():
        for ep in range(args_cli.num_demos):
            if not simulation_app.is_running():
                break
            for t in range(args_cli.episode_len):
                if args_cli.action_mode == "random":
                    action = torch.zeros((1, action_dim), device=device)
                    action[:, :3] = (torch.rand((1, 3), device=device) * 2 - 1) * args_cli.max_delta
                    action[:, 3:6] = (torch.rand((1, 3), device=device) * 2 - 1) * args_cli.max_rot_delta
                    if action_dim >= 7:
                        action[:, 6] = 1.0 if t < args_cli.episode_len // 2 else -1.0
                else:
                    action = _scripted_action(
                        t, args_cli.episode_len, device, args_cli.max_delta, args_cli.max_rot_delta
                    )
                    if action.shape[-1] != action_dim:  # pad/truncate defensively
                        fixed = torch.zeros((1, action_dim), device=device)
                        n = min(action_dim, action.shape[-1])
                        fixed[:, :n] = action[:, :n]
                        action = fixed
                env.step(action)
            # End-of-episode: env.reset() triggers record_pre_reset -> exports this episode (EXPORT_ALL).
            env.reset()
            exported = env.recorder_manager.exported_successful_episode_count + (ep + 1)
            print(f"[gen] episode {ep + 1}/{args_cli.num_demos} recorded.", flush=True)

    env.close()
    print(f"[gen] DONE. Synthetic dataset written to: {dataset_file}", flush=True)
    print("[gen] Inspect with adapters/data_conversion/inspect_hdf5.py; do NOT use for real training.", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
